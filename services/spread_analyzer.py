import httpx
from typing import List, Tuple, Dict, Any
from datetime import datetime
from collections import Counter
import numpy as np
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from sklearn.cluster import KMeans, DBSCAN
from models.schemas import SpreadPoint, APIResponse


def _odd_int(n: int) -> int:
    """Повертає найближче непарне число <= n (мінімум 1)."""
    n_int = int(n)
    if n_int <= 1:
        return 1
    return n_int if (n_int % 2 == 1) else (n_int - 1)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Просте згладжування ковзним середнім."""
    w = max(1, int(window))
    if w == 1:
        return values.copy()
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(values, kernel, mode="same")


def _smooth_series(values: List[float]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Згладжує ряд для стабільнішого пошуку екстремумів.
    Дефолт: Savitzky–Golay; fallback: moving average.
    """
    arr = np.array(values, dtype=float)
    n = int(arr.size)
    if n == 0:
        return arr, {"method": "none", "reason": "empty"}

    # Обробка NaN/inf (замінюємо лінійною інтерполяцією, якщо можливо)
    finite_mask = np.isfinite(arr)
    if not np.all(finite_mask):
        finite_idx = np.where(finite_mask)[0]
        if finite_idx.size >= 2:
            arr_interp = arr.copy()
            missing_idx = np.where(~finite_mask)[0]
            arr_interp[missing_idx] = np.interp(missing_idx, finite_idx, arr[finite_idx])
            arr = arr_interp
        else:
            # Якщо даних замало — підміняємо не-фінітні нулями
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Savitzky–Golay: window_length має бути непарним і > polyorder
    polyorder = 3
    # Евристика: ~5% ряду, але в межах [9, 101]
    proposed = max(9, min(101, int(round(n * 0.05))))
    window_length = _odd_int(proposed)
    if window_length <= polyorder:
        window_length = _odd_int(polyorder + 2)
    if window_length > n:
        window_length = _odd_int(n)
    if window_length <= polyorder or window_length < 5:
        # fallback на moving average
        w = max(3, _odd_int(max(3, int(round(n * 0.03)))))
        if w > n:
            w = _odd_int(n)
        smoothed = _moving_average(arr, w)
        return smoothed, {"method": "moving_average", "window": int(w)}

    try:
        smoothed = savgol_filter(arr, window_length=window_length, polyorder=polyorder, mode="interp")
        return smoothed, {"method": "savgol", "window_length": int(window_length), "polyorder": int(polyorder)}
    except Exception as e:
        w = max(3, _odd_int(max(3, int(round(n * 0.03)))))
        if w > n:
            w = _odd_int(n)
        smoothed = _moving_average(arr, w)
        return smoothed, {"method": "moving_average", "window": int(w), "fallback_reason": str(e)}


def _compute_level_from_peaks(
    values: List[float],
    *,
    quantile_among_peaks: float,
    fallback_quantile: float,
    top_k_cap: int = 30,
) -> Tuple[float | None, Dict[str, Any]]:
    """
    Обчислює рівень за локальними максимумами:
    - згладжує ряд
    - знаходить піки на згладженому
    - бере значення оригінального ряду в піках
    - відбирає верхній хвіст (quantile_among_peaks) + cap top-K
    - агрегує медіаною і кліпує по q01–q99 сирого ряду
    """
    debug: Dict[str, Any] = {}
    n = int(len(values)) if values else 0
    if n < 3:
        return (float(values[-1]) if n > 0 else None), {"reason": "too_few_points", "n": n}

    smoothed, smooth_dbg = _smooth_series(values)
    debug["smoothing"] = smooth_dbg

    # Евристики для find_peaks
    # distance: щоб не брати сусідні шумові піки
    distance = max(2, int(round(n / 50)))  # ~2% довжини
    # prominence: робастно від розкиду
    q25, q75 = np.percentile(smoothed, [25, 75])
    iqr = float(q75 - q25)
    std = float(np.std(smoothed))
    prominence = max(1e-6, 0.15 * iqr, 0.10 * std)

    peaks_idx, props = find_peaks(smoothed, distance=distance, prominence=prominence)
    peaks_idx = peaks_idx.astype(int).tolist() if hasattr(peaks_idx, "astype") else list(peaks_idx)
    debug["peaks"] = {
        "count": int(len(peaks_idx)),
        "distance": int(distance),
        "prominence": float(prominence),
    }

    arr = np.array(values, dtype=float)
    finite_mask = np.isfinite(arr)
    if not np.all(finite_mask):
        arr = np.nan_to_num(arr, nan=np.nan, posinf=np.nan, neginf=np.nan)

    peak_values = [float(arr[i]) for i in peaks_idx if 0 <= i < n and np.isfinite(arr[i])]
    debug["peak_values_count"] = int(len(peak_values))

    if len(peak_values) >= 3:
        thr = float(np.quantile(peak_values, quantile_among_peaks))
        candidates = [v for v in peak_values if v >= thr]
        # cap top-K за значенням
        candidates_sorted = sorted(candidates, reverse=True)[: max(1, min(int(top_k_cap), len(candidates)))]
        candidates = candidates_sorted
        debug["candidate_selection"] = {
            "quantile_among_peaks": float(quantile_among_peaks),
            "threshold": float(thr),
            "top_k_cap": int(top_k_cap),
            "candidates_count": int(len(candidates)),
        }
    else:
        candidates = []
        debug["candidate_selection"] = {"reason": "insufficient_peaks", "needed": 3}

    # fallback: квантиль по всьому ряду
    if not candidates:
        series_finite = arr[np.isfinite(arr)]
        if series_finite.size == 0:
            return None, {**debug, "fallback": {"reason": "no_finite_values"}}
        level = float(np.quantile(series_finite, fallback_quantile))
        debug["fallback"] = {"method": "series_quantile", "q": float(fallback_quantile), "value": float(level)}
        return level, debug

    level_raw = float(np.median(np.array(candidates, dtype=float)))
    # кліп по q01–q99 сирого ряду
    series_finite = arr[np.isfinite(arr)]
    q01 = float(np.quantile(series_finite, 0.01)) if series_finite.size > 0 else level_raw
    q99 = float(np.quantile(series_finite, 0.99)) if series_finite.size > 0 else level_raw
    level = float(np.clip(level_raw, q01, q99))
    debug["aggregation"] = {
        "method": "median",
        "level_raw": float(level_raw),
        "clip_q01": float(q01),
        "clip_q99": float(q99),
        "level": float(level),
    }
    debug["candidates_sample"] = candidates[:10]
    return level, debug


def _format_dt(ts: int) -> str:
    """Форматує unix timestamp (секунди) у читабельний рядок."""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _filter_by_lookback_hours(spreads: List[SpreadPoint], hours: int, end_ts: int) -> List[SpreadPoint]:
    """Фільтрує точки за останні `hours` годин відносно `end_ts`."""
    start_ts = int(end_ts - hours * 3600)
    return [p for p in spreads if start_ts <= p.t <= end_ts]


def _series_basic_stats(values: List[float]) -> Dict[str, Any]:
    """Базова статистика для ряду."""
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "avg": None,
            "min_abs": None,
            "max_abs": None,
        }

    vmin = float(min(values))
    vmax = float(max(values))
    avg = float(sum(values) / len(values))
    abs_values = [abs(v) for v in values]
    return {
        "count": int(len(values)),
        "min": vmin,
        "max": vmax,
        "avg": avg,
        "min_abs": float(min(abs_values)) if abs_values else None,
        "max_abs": float(max(abs_values)) if abs_values else None,
    }


def _series_crosses_zero(values: List[float]) -> bool:
    """Перевіряє, чи ряд перетинав/торкався 0 (наявні значення різних знаків або 0)."""
    if not values:
        return False
    has_neg = any(v < 0 for v in values)
    has_pos = any(v > 0 for v in values)
    has_zero = any(v == 0 for v in values)
    return bool(has_zero or (has_neg and has_pos))


def _series_reaches_abs_threshold(values: List[float], threshold: float) -> bool:
    """Перевіряє, чи було значення з модулем <= threshold."""
    if not values:
        return False
    return any(abs(v) <= threshold for v in values)


def compute_lookback_checks(
    spreads: List[SpreadPoint],
    hours_list: List[int] = None,
    abs_thresholds: List[float] = None,
) -> Dict[str, Any]:
    """
    Обчислює min/max/avg та перевірки перетину 0 і досягнення |x|<=threshold
    для серій in та out за заданими часовими вікнами.
    """
    if hours_list is None:
        hours_list = [2, 4, 8, 16]
    if abs_thresholds is None:
        abs_thresholds = [0.5, 1.0]

    if not spreads:
        return {"hours": hours_list, "abs_thresholds": abs_thresholds, "windows": {}}

    sorted_spreads = sorted(spreads, key=lambda x: x.t)
    end_ts = int(sorted_spreads[-1].t)

    windows: Dict[str, Any] = {}
    for hours in hours_list:
        window_spreads = _filter_by_lookback_hours(sorted_spreads, hours, end_ts)
        values_in = [p.in_ for p in window_spreads]
        values_out = [p.out for p in window_spreads]

        reached_in = {str(th): _series_reaches_abs_threshold(values_in, th) for th in abs_thresholds}
        reached_out = {str(th): _series_reaches_abs_threshold(values_out, th) for th in abs_thresholds}

        start_ts = int(end_ts - hours * 3600)
        windows[str(hours)] = {
            "hours": int(hours),
            "start_ts": start_ts,
            "end_ts": end_ts,
            "start_time": _format_dt(start_ts),
            "end_time": _format_dt(end_ts),
            "points": int(len(window_spreads)),
            "in": {
                "stats": _series_basic_stats(values_in),
                "crossed_zero": _series_crosses_zero(values_in),
                "reached_abs": reached_in,
            },
            "out": {
                "stats": _series_basic_stats(values_out),
                "crossed_zero": _series_crosses_zero(values_out),
                "reached_abs": reached_out,
            },
        }

    return {
        "hours": [int(h) for h in hours_list],
        "abs_thresholds": [float(t) for t in abs_thresholds],
        "end_ts": end_ts,
        "end_time": _format_dt(end_ts),
        "windows": windows,
    }


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


def _analyze_values_histogram(values: List[float], bins: int = None) -> Dict[str, Any]:
    """Аналізує ряд через гістограму з бінінгом."""
    if not values:
        return {
            "most_frequent_bin": None,
            "frequency": 0,
            "bins": [],
            "frequencies": [],
            "bin_edges": [],
            "top_bins": [],
        }

    arr = np.array(values)

    # Автоматичне визначення кількості бінів за правилом Стреджеса
    if bins is None:
        n = len(arr)
        bins = int(np.ceil(1 + np.log2(n))) if n > 0 else 10

    frequencies, bin_edges = np.histogram(arr, bins=bins)

    max_freq_idx = int(np.argmax(frequencies)) if len(frequencies) > 0 else 0
    most_frequent_bin = (float(bin_edges[max_freq_idx]), float(bin_edges[max_freq_idx + 1])) if len(bin_edges) > 1 else None
    frequency = int(frequencies[max_freq_idx]) if len(frequencies) > 0 else 0

    # Топ-5 найчастіших бінів
    top_bins: List[Dict[str, Any]] = []
    if len(frequencies) > 0:
        top_indices = np.argsort(frequencies)[::-1][:5]
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


def analyze_out_histogram(spreads: List[SpreadPoint], bins: int = None) -> Dict[str, Any]:
    """
    Аналізує значення out через гістограму з бінінгом.
    
    Args:
        spreads: Список точок спреду
        bins: Кількість бінів (None для автоматичного визначення)
    
    Returns:
        Словник з результатами аналізу гістограми
    """
    values = [point.out for point in spreads] if spreads else []
    return _analyze_values_histogram(values, bins=bins)


def analyze_in_histogram(spreads: List[SpreadPoint], bins: int = None) -> Dict[str, Any]:
    """Аналізує значення in через гістограму з бінінгом."""
    values = [point.in_ for point in spreads] if spreads else []
    return _analyze_values_histogram(values, bins=bins)


def _analyze_values_kde(values: List[float], num_points: int = 1000) -> Dict[str, Any]:
    """Аналізує ряд через KDE (Kernel Density Estimation)."""
    if not values or len(values) < 2:
        return {"peaks": [], "peak_values": [], "density_x": [], "density_y": []}

    arr = np.array(values)
    try:
        kde = stats.gaussian_kde(arr)
    except Exception:
        # Напр. сингулярна коваріація при нульовій дисперсії
        return {"peaks": [], "peak_values": [], "density_x": [], "density_y": []}

    x_min, x_max = float(arr.min()), float(arr.max())
    x_range = x_max - x_min
    if x_range == 0:
        return {"peaks": [x_min], "peak_values": [1.0], "density_x": [x_min], "density_y": [1.0]}

    x_eval = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, num_points)
    density = kde(x_eval)

    height_threshold = float(np.max(density)) * 0.1 if len(density) > 0 else 0.0
    peaks_indices, _properties = find_peaks(density, height=height_threshold, distance=max(1, num_points // 20))

    peaks = x_eval[peaks_indices].tolist() if len(peaks_indices) > 0 else []
    peak_values = density[peaks_indices].tolist() if len(peaks_indices) > 0 else []

    # Сортуємо піки за значенням щільності (найвищі першими)
    if peaks:
        sorted_indices = np.argsort(peak_values)[::-1]
        peaks = [peaks[i] for i in sorted_indices]
        peak_values = [peak_values[i] for i in sorted_indices]

    return {"peaks": peaks, "peak_values": peak_values, "density_x": x_eval.tolist(), "density_y": density.tolist()}


def analyze_out_kde(spreads: List[SpreadPoint], num_points: int = 1000) -> Dict[str, Any]:
    """
    Аналізує значення out через KDE (Kernel Density Estimation).
    
    Args:
        spreads: Список точок спреду
        num_points: Кількість точок для оцінки щільності
    
    Returns:
        Словник з результатами KDE аналізу
    """
    values = [point.out for point in spreads] if spreads else []
    return _analyze_values_kde(values, num_points=num_points)


def analyze_in_kde(spreads: List[SpreadPoint], num_points: int = 1000) -> Dict[str, Any]:
    """Аналізує значення in через KDE (Kernel Density Estimation)."""
    values = [point.in_ for point in spreads] if spreads else []
    return _analyze_values_kde(values, num_points=num_points)


def _analyze_values_mode(values: List[float], precision: float = 0.01) -> Dict[str, Any]:
    """Знаходить моду (найчастіше значення) для ряду з округленням до precision."""
    if not values:
        return {"value": None, "frequency": 0}

    rounded_values = [round(v / precision) * precision for v in values]
    counter = Counter(rounded_values)
    most_common = counter.most_common(1)[0]
    return {"value": most_common[0], "frequency": most_common[1]}


def analyze_out_mode(spreads: List[SpreadPoint], precision: float = 0.01) -> Dict[str, Any]:
    """
    Знаходить моду (найчастіше значення) для out.
    
    Args:
        spreads: Список точок спреду
        precision: Точність округлення для групування значень
    
    Returns:
        Словник з результатами моди
    """
    values = [point.out for point in spreads] if spreads else []
    return _analyze_values_mode(values, precision=precision)


def analyze_in_mode(spreads: List[SpreadPoint], precision: float = 0.01) -> Dict[str, Any]:
    """Знаходить моду (найчастіше значення) для in."""
    values = [point.in_ for point in spreads] if spreads else []
    return _analyze_values_mode(values, precision=precision)


def _analyze_values_clustering(values: List[float], method: str = "kmeans", n_clusters: int = None) -> Dict[str, Any]:
    """Аналізує ряд через кластеризацію (KMeans або DBSCAN)."""
    if not values or len(values) < 2:
        return {"clusters": []}

    arr = np.array(values).reshape(-1, 1)

    if method == "kmeans":
        if n_clusters is None:
            n_clusters = min(5, max(2, len(values) // 100))
        n_clusters = max(1, min(int(n_clusters), len(values)))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(arr)
        centers = kmeans.cluster_centers_.flatten()

        clusters: List[Dict[str, Any]] = []
        for i, center in enumerate(centers):
            cluster_points = arr[labels == i].flatten()
            if len(cluster_points) > 0:
                clusters.append({
                    "center": float(center),
                    "points": int(len(cluster_points)),
                    "range": (float(cluster_points.min()), float(cluster_points.max()))
                })
        clusters.sort(key=lambda x: x["points"], reverse=True)
        return {"clusters": clusters, "method": "kmeans"}

    # dbscan
    sorted_values = np.sort(arr.flatten())
    diffs = np.diff(sorted_values)
    diffs_pos = diffs[diffs > 0]
    if len(diffs_pos) == 0:
        eps = 1e-6
    else:
        eps = float(np.percentile(diffs_pos, 50)) * 3
        if eps <= 0:
            eps = 1e-6

    dbscan = DBSCAN(eps=eps, min_samples=max(2, len(values) // 100))
    labels = dbscan.fit_predict(arr)

    clusters: List[Dict[str, Any]] = []
    unique_labels = set(labels)
    unique_labels.discard(-1)  # шум

    for label in unique_labels:
        cluster_points = arr[labels == label].flatten()
        if len(cluster_points) > 0:
            clusters.append({
                "center": float(cluster_points.mean()),
                "points": int(len(cluster_points)),
                "range": (float(cluster_points.min()), float(cluster_points.max()))
            })
    clusters.sort(key=lambda x: x["points"], reverse=True)
    return {"clusters": clusters, "method": "dbscan"}


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
    values = [point.out for point in spreads] if spreads else []
    return _analyze_values_clustering(values, method=method, n_clusters=n_clusters)


def analyze_in_clustering(spreads: List[SpreadPoint], method: str = "kmeans", n_clusters: int = None) -> Dict[str, Any]:
    """Аналізує значення in через кластеризацію."""
    values = [point.in_ for point in spreads] if spreads else []
    return _analyze_values_clustering(values, method=method, n_clusters=n_clusters)


def _analyze_values_quantiles(values: List[float]) -> Dict[str, Any]:
    """Обчислює квантилі для ряду."""
    if not values:
        return {"median": None, "q1": None, "q3": None, "iqr": None}

    arr = np.array(values)
    median = float(np.median(arr))
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1
    return {"median": median, "q1": q1, "q3": q3, "iqr": iqr}


def analyze_out_quantiles(spreads: List[SpreadPoint]) -> Dict[str, Any]:
    """
    Обчислює квантилі для значень out.
    
    Args:
        spreads: Список точок спреду
    
    Returns:
        Словник з квантилями
    """
    values = [point.out for point in spreads] if spreads else []
    return _analyze_values_quantiles(values)


def analyze_in_quantiles(spreads: List[SpreadPoint]) -> Dict[str, Any]:
    """Обчислює квантилі для значень in."""
    values = [point.in_ for point in spreads] if spreads else []
    return _analyze_values_quantiles(values)


def calculate_average_from_methods(
    out_analysis: Dict[str, Any], 
    enabled_methods: List[str] = None
) -> Dict[str, Any]:
    """
    Обчислює середнє значення з вибраних методик аналізу.
    
    Args:
        out_analysis: Словник з результатами аналізу
        enabled_methods: Список назв методик для включення. 
                         Якщо None, використовуються всі доступні методики.
                         Можливі значення: "histogram", "mode", "kde", "clustering", "quantiles"
    
    Returns:
        Словник з середніми значеннями та деталями
    """
    if enabled_methods is None:
        enabled_methods = ["histogram", "mode", "kde", "clustering", "quantiles"]
    
    values = []
    method_names = []
    
    # Гістограма - центр найчастішого біну
    if "histogram" in enabled_methods and out_analysis.get("histogram", {}).get("most_frequent_bin"):
        bin_range = out_analysis["histogram"]["most_frequent_bin"]
        center = (bin_range[0] + bin_range[1]) / 2
        values.append(center)
        method_names.append("Гістограма")
    
    # Мода
    if "mode" in enabled_methods and out_analysis.get("mode", {}).get("value") is not None:
        values.append(out_analysis["mode"]["value"])
        method_names.append("Мода")
    
    # KDE - найвищий пік
    if "kde" in enabled_methods and out_analysis.get("kde", {}).get("peaks") and len(out_analysis["kde"]["peaks"]) > 0:
        values.append(out_analysis["kde"]["peaks"][0])  # Найвищий пік
        method_names.append("KDE (найвищий пік)")
    
    # Кластеризація - центр найбільшого кластера
    if "clustering" in enabled_methods and out_analysis.get("clustering", {}).get("clusters") and len(out_analysis["clustering"]["clusters"]) > 0:
        values.append(out_analysis["clustering"]["clusters"][0]["center"])  # Найбільший кластер
        method_names.append("Кластеризація")
    
    # Квантилі - медіана
    if "quantiles" in enabled_methods and out_analysis.get("quantiles", {}).get("median") is not None:
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


def calculate_simple_mean(spreads: List[SpreadPoint]) -> float:
    """
    Обчислює просте середнє арифметичне всіх значень out.
    
    Args:
        spreads: Список точок спреду
    
    Returns:
        Середнє арифметичне значення out
    """
    if not spreads:
        return 0.0
    
    out_values = [point.out for point in spreads]
    return float(np.mean(out_values))


def calculate_simple_mean_in(spreads: List[SpreadPoint]) -> float:
    """
    Обчислює просте середнє арифметичне всіх значень in.
    """
    if not spreads:
        return 0.0

    in_values = [point.in_ for point in spreads]
    return float(np.mean(in_values))


def prepare_chart_data(
    spreads: List[SpreadPoint],
    avg_method: str = "methods",
    enabled_methods: List[str] = None
) -> Dict[str, Any]:
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
        empty_analysis = {
            "histogram": _analyze_values_histogram([]),
            "kde": _analyze_values_kde([]),
            "mode": _analyze_values_mode([]),
            "clustering": _analyze_values_clustering([]),
            "quantiles": _analyze_values_quantiles([]),
        }
        return {
            "timestamps": [],
            "in_values": [],
            "out_values": [],
            "current_in": None,
            "current_out": None,
            "total_points": 0,
            "time_range": "Немає даних",
            "lookback_checks": {"hours": [2, 4, 8, 16], "abs_thresholds": [0.5, 1.0], "windows": {}},
            "out_analysis": empty_analysis,
            "in_analysis": empty_analysis,
            "entry_in_level": None,
            "exit_out_level": None,
            "levels_debug": {"in": {"reason": "no_data"}, "out": {"reason": "no_data"}},
            "average_from_methods": {
                "average": None,
                "count": 0,
                "values": [],
                "method_names": []
            },
            "average_in_from_methods": {
                "average": None,
                "count": 0,
                "values": [],
                "method_names": []
            },
            "simple_mean": 0.0,
            "simple_mean_in": 0.0
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

    # Аналіз найчастіших значень in
    in_analysis = {
        "histogram": analyze_in_histogram(sorted_spreads),
        "kde": analyze_in_kde(sorted_spreads),
        "mode": analyze_in_mode(sorted_spreads),
        "clustering": analyze_in_clustering(sorted_spreads),
        "quantiles": analyze_in_quantiles(sorted_spreads)
    }
    
    # Обчислюємо середнє значення залежно від вибраного методу
    simple_mean = calculate_simple_mean(sorted_spreads)  # out
    simple_mean_in = calculate_simple_mean_in(sorted_spreads)
    
    if avg_method == "simple":
        # Використовуємо просте середнє арифметичне
        average_from_methods = {
            "average": simple_mean,
            "count": 1,
            "values": [simple_mean],
            "method_names": ["Просте середнє арифметичне"],
            "min": simple_mean,
            "max": simple_mean,
            "std": 0.0
        }
        average_in_from_methods = {
            "average": simple_mean_in,
            "count": 1,
            "values": [simple_mean_in],
            "method_names": ["Просте середнє арифметичне"],
            "min": simple_mean_in,
            "max": simple_mean_in,
            "std": 0.0
        }
    elif avg_method == "both":
        # Обчислюємо середнє з методик та додаємо просте середнє
        methods_avg = calculate_average_from_methods(out_analysis, enabled_methods)
        methods_avg_in = calculate_average_from_methods(in_analysis, enabled_methods)
        if methods_avg["average"] is not None:
            all_values = methods_avg["values"] + [simple_mean]
            all_names = methods_avg["method_names"] + ["Просте середнє"]
            average_from_methods = {
                "average": float(np.mean(all_values)),
                "count": len(all_values),
                "values": all_values,
                "method_names": all_names,
                "min": float(np.min(all_values)),
                "max": float(np.max(all_values)),
                "std": float(np.std(all_values)) if len(all_values) > 1 else 0.0
            }
        else:
            # Якщо методики не дали результат, використовуємо просте середнє
            average_from_methods = {
                "average": simple_mean,
                "count": 1,
                "values": [simple_mean],
                "method_names": ["Просте середнє арифметичне"],
                "min": simple_mean,
                "max": simple_mean,
                "std": 0.0
            }

        if methods_avg_in["average"] is not None:
            all_values_in = methods_avg_in["values"] + [simple_mean_in]
            all_names_in = methods_avg_in["method_names"] + ["Просте середнє"]
            average_in_from_methods = {
                "average": float(np.mean(all_values_in)),
                "count": len(all_values_in),
                "values": all_values_in,
                "method_names": all_names_in,
                "min": float(np.min(all_values_in)),
                "max": float(np.max(all_values_in)),
                "std": float(np.std(all_values_in)) if len(all_values_in) > 1 else 0.0
            }
        else:
            average_in_from_methods = {
                "average": simple_mean_in,
                "count": 1,
                "values": [simple_mean_in],
                "method_names": ["Просте середнє арифметичне"],
                "min": simple_mean_in,
                "max": simple_mean_in,
                "std": 0.0
            }
    else:  # avg_method == "methods" (за замовчуванням)
        # Використовуємо тільки методики
        average_from_methods = calculate_average_from_methods(out_analysis, enabled_methods)
        average_in_from_methods = calculate_average_from_methods(in_analysis, enabled_methods)

    lookback_checks = compute_lookback_checks(sorted_spreads)

    # Рівні входу/виходу (A+C): згладжування → піки → робастна агрегація
    entry_in_level, entry_in_dbg = _compute_level_from_peaks(
        in_values,
        quantile_among_peaks=0.80,
        fallback_quantile=0.90,
        top_k_cap=30,
    )
    exit_out_level, exit_out_dbg = _compute_level_from_peaks(
        out_values,
        quantile_among_peaks=0.80,
        fallback_quantile=0.80,
        top_k_cap=30,
    )
    levels_debug = {"in": entry_in_dbg, "out": exit_out_dbg}
    
    return {
        "timestamps": timestamps,
        "in_values": in_values,
        "out_values": out_values,
        "current_in": current_in,
        "current_out": current_out,
        "last_timestamp": last_timestamp,
        "total_points": len(spreads),
        "time_range": time_range,
        "lookback_checks": lookback_checks,
        "out_analysis": out_analysis,
        "in_analysis": in_analysis,
        "entry_in_level": entry_in_level,
        "exit_out_level": exit_out_level,
        "levels_debug": levels_debug,
        "average_from_methods": average_from_methods,
        "average_in_from_methods": average_in_from_methods,
        "simple_mean": simple_mean,
        "simple_mean_in": simple_mean_in
    }

