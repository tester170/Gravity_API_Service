"""FastAPI-сервис применения Ridge-модели к подготовленным признакам рельефа."""

from contextlib import asynccontextmanager  # Средство загрузки модели при запуске сервера.
from typing import Literal  # Ограничение формата ответа двумя вариантами.
import pandas as pd  # Библиотека для преобразования строк запроса в таблицу.
from fastapi import FastAPI, HTTPException, Request  # Сервис, HTTP-ошибка и текущий запрос.
from fastapi.responses import HTMLResponse  # Ответ с готовой HTML-картой.
from gravity_backend import load_model, predict_gravity  # Загрузка модели и расчёт.
from gravity_map import build_gravity_map  # Построение интерактивной карты.


@asynccontextmanager  # Выполняем подготовку до начала обработки запросов.
async def lifespan(app):
    app.state.model = load_model()  # Загружаем модель один раз на серверный процесс.
    yield  # Сервер принимает запросы с уже загруженной моделью.


app = FastAPI(lifespan=lifespan)  # Создаём сервис; документация доступна по адресу /docs.


@app.post("/predict")  # Принимаем данные методом POST, а не через адресную строку.
def predict(rows: list[dict[str, float]], request: Request, output: Literal["json", "html"] = "json"):
    """Передать список строк с координатами и девятью признаками из terrain_demo.csv."""
    try:  # Преобразуем ошибки входных данных в понятный ответ API.
        result = predict_gravity(pd.DataFrame(rows), request.app.state.model)  # Выполняем расчёт.
    except ValueError as error:  # Не скрываем ошибки данных за общим сообщением сервера.
        raise HTTPException(status_code=422, detail=str(error)) from error
    if output == "html":  # По запросу пользователя возвращаем готовую карту.
        return HTMLResponse(build_gravity_map(result))  # Браузер получает полноценную HTML-страницу.
    return result.to_dict(orient="records")  # По умолчанию возвращаем список расчётных строк в JSON.
