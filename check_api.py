"""Пример обращения к API: расчёт, сохранение CSV и открытие интерактивной карты."""

from pathlib import Path  # Класс для работы с файлами.
import webbrowser  # Средство открытия карты в браузере.
import pandas as pd  # Библиотека для чтения и сохранения таблиц.
import requests  # Библиотека для HTTP-запросов к FastAPI.

PROJECT_DIR = Path(__file__).resolve().parent  # Папка клиентского примера.
API_URL = "http://127.0.0.1:8001/predict"  # Здесь можно указать адрес удалённого сервера.


def main():
    """Отправить демонстрационные данные и сохранить оба формата результата."""
    data = pd.read_csv(PROJECT_DIR / "data" / "terrain_demo.csv")  # Читаем признаки без опорного поля.
    rows = data.to_dict(orient="records")  # Превращаем строки таблицы в список словарей для JSON.
    response = requests.post(API_URL, json=rows, timeout=60)  # Запрашиваем численные результаты.
    response.raise_for_status()  # Не принимаем ошибку сервера за успешный расчёт.
    result = pd.DataFrame(response.json())  # Преобразуем ответ API обратно в таблицу.
    output_dir = PROJECT_DIR / "results"  # Все результаты клиента храним рядом с проектом.
    output_dir.mkdir(exist_ok=True)  # Создаём папку при первом запуске.
    result.to_csv(output_dir / "terrain_predictions.csv", index=False)  # Сохраняем расчёт без индекса.

    response = requests.post(API_URL, params={"output": "html"}, json=rows, timeout=60)  # Запрашиваем карту.
    response.raise_for_status()  # Проверяем успешность второго запроса.
    map_file = output_dir / "terrain_map.html"  # Имя самостоятельной интерактивной карты.
    map_file.write_text(response.text, encoding="utf-8")  # Сохраняем HTML с русскими подписями.
    print(result.head().to_string(index=False))  # Показываем первые пять строк расчёта.
    print(f"Обработано точек: {len(result)}. Результаты: {output_dir}")  # Сообщаем, где лежат файлы.
    webbrowser.open(map_file.as_uri())  # Открываем результат в браузере.


if __name__ == "__main__":  # Не отправляем запросы при импорте модуля.
    main()  # Запускаем клиентский пример.
