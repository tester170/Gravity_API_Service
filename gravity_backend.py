"""Уточнение расчёта гравитационного влияния рельефа с помощью Ridge."""

from pathlib import Path  # Класс для работы с путями к файлам.
import joblib  # Библиотека для загрузки модели вместе со стандартизацией.
import numpy as np  # Библиотека для вычислений и проверки чисел.
import pandas as pd  # Библиотека для работы с таблицами.

PROJECT_DIR = Path(__file__).resolve().parent  # Папка проекта.
MODEL_FILE = PROJECT_DIR / "models" / "terrain_model.joblib"  # Подготовленная Ridge-модель.
DATA_FILE = PROJECT_DIR / "data" / "terrain_demo.csv"  # Пример с Северного Кавказа.


def load_model():
    """Загрузить модель и сведения о её входных признаках."""
    return joblib.load(MODEL_FILE)  # Загружаем только доверенный файл из проекта.


def predict_gravity(data, model_bundle):
    """Рассчитать исходное и уточнённое значения для каждой строки таблицы."""
    features = model_bundle["features"]  # Сохраняем порядок признаков при обучении.
    required = ["latitude", "longitude", *features]  # Координаты нужны для карты.
    missing = [name for name in required if name not in data.columns]  # Ищем отсутствующие поля.
    if missing or data.empty:  # Не выполняем расчёт для неполного или пустого набора.
        raise ValueError(f"Нужна непустая таблица. Отсутствующие столбцы: {missing}")
    values = data[required].apply(pd.to_numeric, errors="raise")  # Преобразуем значения в числа.
    if not np.isfinite(values.to_numpy()).all():  # Исключаем пропуски и бесконечности.
        raise ValueError("Входные данные должны содержать только конечные числа без пропусков.")
    if not (values.latitude.between(-90, 90) & values.longitude.between(-180, 180)).all():
        raise ValueError("Широта должна быть от −90 до 90°, долгота — от −180 до 180°.")

    result = values[["latitude", "longitude"]].copy()  # Создаём отдельную таблицу результата.
    layer = 2 * np.pi * 6.67430e-11 * 2670 * values["elevation_m"] * 1e5  # Притяжение слоя, мГал.
    result["baseline_mgal"] = (  # Повторяем быструю аппроксимацию из учебного блокнота.
        layer - 0.020 * values["tpi_7"] - 0.010 * values["tpi_31"] - 0.004 * values["tpi_121"]
    )
    result["correction_mgal"] = model_bundle["model"].predict(values[features])  # Оцениваем остаток.
    result["prediction_mgal"] = result["baseline_mgal"] + result["correction_mgal"]  # Уточняем поле.
    if not np.isfinite(result.to_numpy()).all():  # Не возвращаем переполнение как корректный расчёт.
        raise ValueError("Получены неконечные значения. Проверьте величины и единицы входных признаков.")
    return result  # Опорное поле не используется при получении прогноза.


if __name__ == "__main__":  # Этот пример выполняется только при прямом запуске модуля.
    result = predict_gravity(pd.read_csv(DATA_FILE), load_model())  # Рассчитываем демонстрационный набор.
    print(result.head().to_string(index=False))  # Показываем первые пять строк результата.
    print(f"Обработано точек: {len(result)}")  # Показываем объём расчёта.
