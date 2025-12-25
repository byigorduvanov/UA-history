import httpx
from typing import List, Tuple, Dict, Any
from datetime import datetime
from collections import Counter
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from sklearn.cluster import KMeans, DBSCAN
from models.schemas import SpreadPoint, APIResponse


async def fetch_historical_data(url_or_slug: str, limit: int = 1500) -> List[SpreadPoint]:
    """
    Завантажує історичні дані спреду з API.
    
    Args:
        url_or_slug: Повний URL або slug для формування запиту
        limit: Кількість точок для завантаження (за замовчуванням 1500)
    
    Returns:
        Список точок спреду
    """
    # Визначаємо, чи це повний URL чи slug
    if url_or_slug.startswith("http://") or url_or_slug.startswith("https://"):
        url = url_or_slug
    else:
        # Якщо це slug, формуємо URL
        url = f"https://uainvest.com.ua/api/arbitrage/historical?slug={url_or_slug}&limit={limit}"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Парсимо відповідь через Pydantic
        api_response = APIResponse(**data)
        return api_response.data.spreads


def find_convergence_points(spreads: List[SpreadPoint], tolerance: float = 0.01) -> List[SpreadPoint]:
    """
    Знаходить точки сходження, де in ≈ out з заданою толерантністю.
    
    Args:
        spreads: Список точок спреду
        tolerance: Толерантність для визначення сходження (за замовчуванням 0.01)
    
    Returns:
        Список точок сходження
    """
    convergence_points = []
    
    for point in spreads:
        # Використовуємо in_ замість in, оскільки in - зарезервоване слово
        diff = abs(point.in_ - point.out)
        if diff <= tolerance:
            convergence_points.append(point)
    
    return convergence_points


def calculate_most_frequent_spread(convergence_points: List[SpreadPoint]) -> Tuple[float, int]:
    """
    Рахує найчастіше значення спреду в точках сходження.
    
    Args:
        convergence_points: Список точок сходження
    
    Returns:
        Кортеж (найчастіше значення, кількість входжень)
    """
    from collections import Counter
    
    if not convergence_points:
        return 0.0, 0
    
    # Витягуємо значення in_ (або out, оскільки вони рівні в точках сходження)
    spread_values = [point.in_ for point in convergence_points]
    
    # Підраховуємо частоту кожного значення
    counter = Counter(spread_values)
    
    # Знаходимо найчастіше значення
    most_common = counter.most_common(1)[0]
    
    return most_common[0], most_common[1]


def analyze_out_histogram(spreads: List[SpreadPoint], bins: int = None) -> Dict[str, Any]:
    """
    Аналізує значення out через гістограму з бінінгом.
    
    Args:
        spreads: Список точок спреду
        bins: Кількість бінів (None для автоматичного визначення)
    
    Returns:
        Словник з результатами аналізу гістограми
    """
    if not spreads:
        return {
            "most_frequent_bin": None,
            "frequency": 0,
            "bins": [],
            "frequencies": [],
            "bin_edges": []
        }
    
    out_values = np.array([point.out for point in spreads])
    
    # Автоматичне визначення кількості бінів за правилом Стреджеса
    if bins is None:
        n = len(out_values)
        bins = int(np.ceil(1 + np.log2(n))) if n > 0 else 10
    
    # Обчислюємо гістограму
    frequencies, bin_edges = np.histogram(out_values, bins=bins)
    
    # Знаходимо бін з найбільшою частотою
    max_freq_idx = np.argmax(frequencies)
    most_frequent_bin = (bin_edges[max_freq_idx], bin_edges[max_freq_idx + 1])
    frequency = int(frequencies[max_freq_idx])
    
    # Знаходимо топ-5 найчастіших бінів
    top_indices = np.argsort(frequencies)[::-1][:5]
    top_bins = []
    for idx in top_indices:
        if frequencies[idx] > 0:
            top_bins.append({
                "bin": (float(bin_edges[idx]), float(bin_edges[idx + 1])),
                "center": float((bin_edges[idx] + bin_edges[idx + 1]) / 2),
                "frequency": int(frequencies[idx])
            })
    
    return {
        "most_frequent_bin": most_frequent_bin,
        "frequency": frequency,
        "bins": bin_edges.tolist(),
        "frequencies": frequencies.tolist(),
        "bin_edges": bin_edges.tolist(),
        "top_bins": top_bins
    }


def analyze_out_kde(spreads: List[SpreadPoint], num_points: int = 1000) -> Dict[str, Any]:
    """
    Аналізує значення out через KDE (Kernel Density Estimation).
    
    Args:
        spreads: Список точок спреду
        num_points: Кількість точок для оцінки щільності
    
    Returns:
        Словник з результатами KDE аналізу
    """
    if not spreads or len(spreads) < 2:
        return {
            "peaks": [],
            "peak_values": [],
            "density_x": [],
            "density_y": []
        }
    
    out_values = np.array([point.out for point in spreads])
    
    # Створюємо KDE
    kde = stats.gaussian_kde(out_values)
    
    # Створюємо діапазон для оцінки
    x_min, x_max = out_values.min(), out_values.max()
    x_range = x_max - x_min
    x_eval = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, num_points)
    density = kde(x_eval)
    
    # Знаходимо локальні максимуми (піки)
    # Використовуємо height для фільтрації малих піків
    height_threshold = np.max(density) * 0.1  # 10% від максимального значення
    peaks_indices, properties = find_peaks(density, height=height_threshold, distance=num_points // 20)
    
    peaks = x_eval[peaks_indices].tolist() if len(peaks_indices) > 0 else []
    peak_values = density[peaks_indices].tolist() if len(peaks_indices) > 0 else []
    
    # Сортуємо піки за значенням щільності (найвищі першими)
    if peaks:
        sorted_indices = np.argsort(peak_values)[::-1]
        peaks = [peaks[i] for i in sorted_indices]
        peak_values = [peak_values[i] for i in sorted_indices]
    
    return {
        "peaks": peaks,
        "peak_values": peak_values,
        "density_x": x_eval.tolist(),
        "density_y": density.tolist()
    }


def analyze_out_mode(spreads: List[SpreadPoint], precision: float = 0.01) -> Dict[str, Any]:
    """
    Знаходить моду (найчастіше значення) для out.
    
    Args:
        spreads: Список точок спреду
        precision: Точність округлення для групування значень
    
    Returns:
        Словник з результатами моди
    """
    if not spreads:
        return {
            "value": None,
            "frequency": 0
        }
    
    # Округлюємо значення для групування
    out_values = [round(point.out / precision) * precision for point in spreads]
    
    # Підраховуємо частоту
    counter = Counter(out_values)
    most_common = counter.most_common(1)[0]
    
    return {
        "value": most_common[0],
        "frequency": most_common[1]
    }


def analyze_out_clustering(spreads: List[SpreadPoint], method: str = "kmeans", n_clusters: int = None) -> Dict[str, Any]:
    """
    Аналізує значення out через кластеризацію.
    
    Args:
        spreads: Список точок спреду
        method: Метод кластеризації ("kmeans" або "dbscan")
        n_clusters: Кількість кластерів для K-means (None для автоматичного)
    
    Returns:
        Словник з результатами кластеризації
    """
    if not spreads or len(spreads) < 2:
        return {
            "clusters": []
        }
    
    out_values = np.array([point.out for point in spreads]).reshape(-1, 1)
    
    if method == "kmeans":
        # Автоматичне визначення кількості кластерів (elbow method спрощений)
        if n_clusters is None:
            n_clusters = min(5, max(2, len(spreads) // 100))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(out_values)
        centers = kmeans.cluster_centers_.flatten()
        
        clusters = []
        for i, center in enumerate(centers):
            cluster_points = out_values[labels == i].flatten()
            if len(cluster_points) > 0:
                clusters.append({
                    "center": float(center),
                    "points": int(len(cluster_points)),
                    "range": (float(cluster_points.min()), float(cluster_points.max()))
                })
        
        # Сортуємо за кількістю точок (найбільші першими)
        clusters.sort(key=lambda x: x["points"], reverse=True)
        
    else:  # dbscan
        # Автоматичне визначення eps
        sorted_values = np.sort(out_values.flatten())
        diffs = np.diff(sorted_values)
        eps = np.percentile(diffs[diffs > 0], 50) * 3  # медіана ненульових різниць * 3
        
        dbscan = DBSCAN(eps=eps, min_samples=max(2, len(spreads) // 100))
        labels = dbscan.fit_predict(out_values)
        
        clusters = []
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Виключаємо шум
        
        for label in unique_labels:
            cluster_points = out_values[labels == label].flatten()
            if len(cluster_points) > 0:
                clusters.append({
                    "center": float(cluster_points.mean()),
                    "points": int(len(cluster_points)),
                    "range": (float(cluster_points.min()), float(cluster_points.max()))
                })
        
        # Сортуємо за кількістю точок
        clusters.sort(key=lambda x: x["points"], reverse=True)
    
    return {
        "clusters": clusters,
        "method": method
    }


def analyze_out_quantiles(spreads: List[SpreadPoint]) -> Dict[str, Any]:
    """
    Обчислює квантилі для значень out.
    
    Args:
        spreads: Список точок спреду
    
    Returns:
        Словник з квантилями
    """
    if not spreads:
        return {
            "median": None,
            "q1": None,
            "q3": None,
            "iqr": None
        }
    
    out_values = np.array([point.out for point in spreads])
    
    median = float(np.median(out_values))
    q1 = float(np.percentile(out_values, 25))
    q3 = float(np.percentile(out_values, 75))
    iqr = q3 - q1
    
    return {
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": iqr
    }


def calculate_average_from_methods(out_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Обчислює середнє значення з усіх методик аналізу.
    
    Args:
        out_analysis: Словник з результатами аналізу
    
    Returns:
        Словник з середніми значеннями та деталями
    """
    values = []
    method_names = []
    
    # Гістограма - центр найчастішого біну
    if out_analysis.get("histogram", {}).get("most_frequent_bin"):
        bin_range = out_analysis["histogram"]["most_frequent_bin"]
        center = (bin_range[0] + bin_range[1]) / 2
        values.append(center)
        method_names.append("Гістограма")
    
    # Мода
    if out_analysis.get("mode", {}).get("value") is not None:
        values.append(out_analysis["mode"]["value"])
        method_names.append("Мода")
    
    # KDE - найвищий пік
    if out_analysis.get("kde", {}).get("peaks") and len(out_analysis["kde"]["peaks"]) > 0:
        values.append(out_analysis["kde"]["peaks"][0])  # Найвищий пік
        method_names.append("KDE (найвищий пік)")
    
    # Кластеризація - центр найбільшого кластера
    if out_analysis.get("clustering", {}).get("clusters") and len(out_analysis["clustering"]["clusters"]) > 0:
        values.append(out_analysis["clustering"]["clusters"][0]["center"])  # Найбільший кластер
        method_names.append("Кластеризація")
    
    # Квантилі - медіана
    if out_analysis.get("quantiles", {}).get("median") is not None:
        values.append(out_analysis["quantiles"]["median"])
        method_names.append("Медіана")
    
    if not values:
        return {
            "average": None,
            "count": 0,
            "values": [],
            "method_names": []
        }
    
    average = float(np.mean(values))
    
    return {
        "average": average,
        "count": len(values),
        "values": [float(v) for v in values],
        "method_names": method_names,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)) if len(values) > 1 else 0.0
    }


def prepare_chart_data(spreads: List[SpreadPoint]) -> Dict[str, Any]:
    """
    Підготовлює дані для відображення на графіку.
    
    Args:
        spreads: Список точок спреду
        
    Returns:
        Словник з даними для шаблону:
        - timestamps: список відформатованих дат/часів
        - in_values: список значень in_
        - out_values: список значень out
        - current_in: поточне значення in_ (останнє)
        - current_out: поточне значення out (останнє)
        - total_points: загальна кількість точок
        - time_range: рядок з діапазоном часу
        - out_analysis: результати аналізу найчастіших значень out
    """
    if not spreads:
        return {
            "timestamps": [],
            "in_values": [],
            "out_values": [],
            "current_in": None,
            "current_out": None,
            "total_points": 0,
            "time_range": "Немає даних",
            "out_analysis": {
                "histogram": {},
                "kde": {},
                "mode": {},
                "clustering": {},
                "quantiles": {}
            },
            "average_from_methods": {
                "average": None,
                "count": 0,
                "values": [],
                "method_names": []
            }
        }
    
    # Сортуємо за timestamp (якщо ще не відсортовано)
    sorted_spreads = sorted(spreads, key=lambda x: x.t)
    
    # Витягуємо дані
    timestamps = []
    in_values = []
    out_values = []
    
    for point in sorted_spreads:
        # Конвертуємо timestamp в читабельний формат
        dt = datetime.fromtimestamp(point.t)
        timestamps.append(dt.strftime("%Y-%m-%d %H:%M:%S"))
        in_values.append(point.in_)
        out_values.append(point.out)
    
    # Поточні значення (останні)
    current_in = sorted_spreads[-1].in_ if sorted_spreads else None
    current_out = sorted_spreads[-1].out if sorted_spreads else None
    
    # Формуємо діапазон часу
    if sorted_spreads:
        first_time = datetime.fromtimestamp(sorted_spreads[0].t).strftime("%Y-%m-%d %H:%M:%S")
        last_time = datetime.fromtimestamp(sorted_spreads[-1].t).strftime("%Y-%m-%d %H:%M:%S")
        time_range = f"{first_time} - {last_time}"
    else:
        time_range = "Немає даних"
    
    # Останній timestamp для пунктирних ліній
    last_timestamp = timestamps[-1] if timestamps else ""
    
    # Виконуємо аналіз найчастіших значень out
    out_analysis = {
        "histogram": analyze_out_histogram(sorted_spreads),
        "kde": analyze_out_kde(sorted_spreads),
        "mode": analyze_out_mode(sorted_spreads),
        "clustering": analyze_out_clustering(sorted_spreads),
        "quantiles": analyze_out_quantiles(sorted_spreads)
    }
    
    # Обчислюємо середнє значення з усіх методик
    average_from_methods = calculate_average_from_methods(out_analysis)
    
    return {
        "timestamps": timestamps,
        "in_values": in_values,
        "out_values": out_values,
        "current_in": current_in,
        "current_out": current_out,
        "last_timestamp": last_timestamp,
        "total_points": len(spreads),
        "time_range": time_range,
        "out_analysis": out_analysis,
        "average_from_methods": average_from_methods
    }

