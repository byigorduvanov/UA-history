from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from typing import Optional, Tuple, List
import sys
import re
from services.spread_analyzer import (
    fetch_historical_data,
    find_convergence_points,
    calculate_most_frequent_spread,
    prepare_chart_data
)

app = FastAPI(title="Spread Convergence Analyzer", description="Аналіз точок сходження історичного спреду")
templates = Jinja2Templates(directory="templates")


def extract_exchange_names(url_or_slug: str) -> Tuple[str, str]:
    """
    Витягує назви бірж з URL або slug.
    
    Args:
        url_or_slug: URL або slug (наприклад, "sqd-mexc-swap-gate-swap")
    
    Returns:
        Кортеж (left_exchange, right_exchange)
    """
    # Якщо це slug, витягуємо назви бірж
    if not (url_or_slug.startswith("http://") or url_or_slug.startswith("https://")):
        slug = url_or_slug
    else:
        # Витягуємо slug з URL
        match = re.search(r'slug=([^&]+)', url_or_slug)
        if match:
            slug = match.group(1)
        else:
            return ("Left", "Right")
    
    # Парсимо slug (формат: symbol-exchange1-type-exchange2-type)
    parts = slug.split('-')
    if len(parts) >= 4:
        # Знаходимо індекси бірж
        exchanges = []
        for i, part in enumerate(parts):
            if part in ['mexc', 'gate', 'binance', 'okx', 'bybit']:
                exchanges.append(part.upper())
        
        if len(exchanges) >= 2:
            return (exchanges[0], exchanges[1])
        elif len(exchanges) == 1:
            return (exchanges[0], "Right")
    
    return ("Left", "Right")


@app.get("/analyze")
async def analyze_spread(
    request: Request,
    url: str = Query(..., description="Повний URL або slug для завантаження даних"),
    limit: int = Query(1500, description="Кількість точок для аналізу"),
    tolerance: float = Query(0.01, description="Толерантність для визначення сходження"),
    format: str = Query("json", description="Формат відповіді: json або html"),
    avg_method: str = Query("methods", description="Метод розрахунку середнього: simple, methods, both"),
    use_histogram: Optional[str] = Query(None, description="Використовувати гістограму"),
    use_mode: Optional[str] = Query(None, description="Використовувати моду"),
    use_kde: Optional[str] = Query(None, description="Використовувати KDE"),
    use_clustering: Optional[str] = Query(None, description="Використовувати кластеризацію"),
    use_quantiles: Optional[str] = Query(None, description="Використовувати квантилі")
):
    """
    Аналізує історичні дані спреду та знаходить точки сходження.
    Може повертати JSON або HTML з графіком.
    """
    try:
        # Завантажуємо дані
        spreads = await fetch_historical_data(url, limit)
        
        # Знаходимо точки сходження
        convergence_points = find_convergence_points(spreads, tolerance)
        
        # Рахуємо найчастіше значення
        most_frequent_value, frequency = calculate_most_frequent_spread(convergence_points)
        
        # Обчислюємо статистику
        if convergence_points:
            spread_values = [point.in_ for point in convergence_points]
            min_value = min(spread_values)
            max_value = max(spread_values)
            avg_value = sum(spread_values) / len(spread_values)
        else:
            min_value = max_value = avg_value = 0.0
        
        # Виводимо результати в консоль
        print("\n" + "="*60)
        print("РЕЗУЛЬТАТИ АНАЛІЗУ ТОЧОК СХОДЖЕННЯ СПРЕДУ")
        print("="*60)
        print(f"Загальна кількість точок спреду: {len(spreads)}")
        print(f"Кількість точок сходження: {len(convergence_points)}")
        print(f"Толерантність: {tolerance}")
        print("-"*60)
        
        if convergence_points:
            print(f"Найчастіше значення спреду: {most_frequent_value:.6f}")
            print(f"Кількість входжень: {frequency}")
            print("-"*60)
            print("Статистика значень спреду в точках сходження:")
            print(f"  Мінімальне: {min_value:.6f}")
            print(f"  Максимальне: {max_value:.6f}")
            print(f"  Середнє: {avg_value:.6f}")
        else:
            print("Точки сходження не знайдено з заданою толерантністю.")
        
        print("="*60 + "\n")
        
        # Якщо формат HTML, повертаємо HTML з графіком
        if format.lower() == "html":
            # Формуємо список увімкнених методик
            # Перевіряємо, чи параметр передано і чи він дорівнює "true"
            enabled_methods: List[str] = []
            if use_histogram and use_histogram.lower() == "true":
                enabled_methods.append("histogram")
            if use_mode and use_mode.lower() == "true":
                enabled_methods.append("mode")
            if use_kde and use_kde.lower() == "true":
                enabled_methods.append("kde")
            if use_clustering and use_clustering.lower() == "true":
                enabled_methods.append("clustering")
            if use_quantiles and use_quantiles.lower() == "true":
                enabled_methods.append("quantiles")
            
            # Якщо не обрано жодної методики, використовуємо всі за замовчуванням
            if not enabled_methods and avg_method in ["methods", "both"]:
                enabled_methods = ["histogram", "mode", "kde", "clustering", "quantiles"]
            
            chart_data = prepare_chart_data(spreads, avg_method=avg_method, enabled_methods=enabled_methods if enabled_methods else None)
            left_exchange, right_exchange = extract_exchange_names(url)
            
            return templates.TemplateResponse("chart.html", {
                "request": request,
                **chart_data,
                "left_exchange": left_exchange,
                "right_exchange": right_exchange,
                "left_status": "online",  # Можна додати реальну перевірку статусу
                "right_status": "online",
                "error": None,
                "url": url,  # Зберігаємо URL для AJAX запитів
                "limit": limit,
                "tolerance": tolerance,
                "avg_method": avg_method,
                "use_histogram": "true" if use_histogram and use_histogram.lower() == "true" else "false",
                "use_mode": "true" if use_mode and use_mode.lower() == "true" else "false",
                "use_kde": "true" if use_kde and use_kde.lower() == "true" else "false",
                "use_clustering": "true" if use_clustering and use_clustering.lower() == "true" else "false",
                "use_quantiles": "true" if use_quantiles and use_quantiles.lower() == "true" else "false"
            })
        
        # Повертаємо JSON відповідь
        if format.lower() == "json":
            # Формуємо список увімкнених методик
            enabled_methods: List[str] = []
            if use_histogram and use_histogram.lower() == "true":
                enabled_methods.append("histogram")
            if use_mode and use_mode.lower() == "true":
                enabled_methods.append("mode")
            if use_kde and use_kde.lower() == "true":
                enabled_methods.append("kde")
            if use_clustering and use_clustering.lower() == "true":
                enabled_methods.append("clustering")
            if use_quantiles and use_quantiles.lower() == "true":
                enabled_methods.append("quantiles")
            
            # Якщо не обрано жодної методики, використовуємо всі за замовчуванням
            if not enabled_methods and avg_method in ["methods", "both"]:
                enabled_methods = ["histogram", "mode", "kde", "clustering", "quantiles"]
            
            chart_data = prepare_chart_data(spreads, avg_method=avg_method, enabled_methods=enabled_methods if enabled_methods else None)
            
            return JSONResponse({
                **chart_data,
                "convergence_points_count": len(convergence_points),
                "most_frequent_spread": most_frequent_value,
                "frequency": frequency,
                "convergence_statistics": {
                    "min": min_value,
                    "max": max_value,
                    "avg": avg_value
                }
            })
        
        # Повертаємо JSON відповідь (старий формат для сумісності)
        return JSONResponse({
            "total_points": len(spreads),
            "convergence_points_count": len(convergence_points),
            "tolerance": tolerance,
            "most_frequent_spread": most_frequent_value,
            "frequency": frequency,
            "statistics": {
                "min": min_value,
                "max": max_value,
                "avg": avg_value
            }
        })
    
    except Exception as e:
        error_msg = f"Помилка при аналізі: {str(e)}"
        print(f"\n❌ {error_msg}\n", file=sys.stderr)
        
        if format.lower() == "html":
            return templates.TemplateResponse("chart.html", {
                "request": request,
                "timestamps": [],
                "in_values": [],
                "out_values": [],
                "current_in": None,
                "current_out": None,
                "last_timestamp": "",
                "total_points": 0,
                "time_range": "Помилка завантаження даних",
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
                },
                "left_exchange": "Left",
                "right_exchange": "Right",
                "left_status": "offline",
                "right_status": "offline",
                "error": error_msg
            })
        
        return JSONResponse({"error": error_msg})


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Кореневий endpoint - відображає форму для введення slug"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api")
async def api_info():
    """API інформація endpoint"""
    return JSONResponse({
        "message": "Spread Convergence Analyzer API",
        "endpoints": {
            "/": "Головна сторінка з формою",
            "/analyze": "Аналіз точок сходження спреду (GET параметри: url, limit, tolerance, format). format може бути 'json' або 'html'"
        }
    })
