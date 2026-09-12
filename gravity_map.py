"""Интерактивная карта исходного поля, Ridge-коррекции и итогового расчёта."""

import numpy as np  # Библиотека для расчёта масштаба карты.
import plotly.graph_objects as go  # Библиотека для интерактивных графиков и карт.


def build_gravity_map(data):
    """Построить три переключаемых слоя и вернуть самостоятельную HTML-страницу."""
    fields = ["baseline_mgal", "prediction_mgal", "correction_mgal"]  # Столбцы для трёх слоёв.
    titles = ["Исходный расчёт", "После коррекции Ridge", "Коррекция модели"]  # Подписи переключателя.
    lower = data[fields[:2]].min().min()  # Общая нижняя граница шкалы исходного и итогового полей.
    upper = data[fields[:2]].max().max()  # Общая верхняя граница для честного сравнения.
    limit = max(data["correction_mgal"].abs().max(), 0.01)  # Симметричная шкала коррекции вокруг нуля.
    latitude_scale = max(np.cos(np.deg2rad(data.latitude.mean())), 0.01)  # Учитываем широту в проекции карты.
    span = max((data.latitude.max() - data.latitude.min()) / latitude_scale,
               data.longitude.max() - data.longitude.min(), 0.01)  # Охват участка с учётом проекции.
    zoom = float(np.clip(np.log2(360 / span), 1, 14))  # Участок занимает около 512 пикселей по большей стороне.
    figure = go.Figure()  # Создаём пустую карту.

    for index, (field, title) in enumerate(zip(fields, titles)):  # Добавляем три слоя точек.
        is_correction = field == "correction_mgal"  # Для поправочного остатка нужна отдельная шкала.
        figure.add_trace(go.Scattermapbox(  # Координаты описывают центры ячеек исходного растра.
            lat=data["latitude"].tolist(), lon=data["longitude"].tolist(),
            mode="markers", name=title, visible=index == 1,
            marker={"size": 7, "opacity": 0.9, "color": data[field].tolist(),
                    "colorscale": "RdBu_r" if is_correction else "Viridis",
                    "cmin": -limit if is_correction else lower,
                    "cmax": limit if is_correction else upper,
                    "showscale": True, "colorbar": {"title": "мГал", "thickness": 14}},
            customdata=data[fields].to_numpy().tolist(),  # Передаём в подсказку все три значения.
            hovertemplate=("Широта: %{lat:.5f}°<br>Долгота: %{lon:.5f}°<br>"
                           "Исходное: %{customdata[0]:.2f} мГал<br>"
                           "После Ridge: %{customdata[1]:.2f} мГал<br>"
                           "Коррекция: %{customdata[2]:.2f} мГал<extra></extra>"),
        ))

    buttons = [  # Каждый пункт показывает только выбранный слой.
        {"label": title, "method": "update", "args": [{"visible": [j == i for j in range(3)]}]}
        for i, title in enumerate(titles)
    ]
    figure.update_layout(  # Оформляем карту штатными параметрами Plotly.
        title="Гравитационное влияние рельефа · расчёт по подготовленным признакам",
        mapbox={"style": "white-bg", "zoom": zoom,  # Строим точки, не ожидая загрузки подложки.
                "center": {"lat": data.latitude.mean(), "lon": data.longitude.mean()},
                "layers": [{"sourcetype": "raster", "below": "traces",  # Подложка находится под точками.
                            "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
                            "sourceattribution": "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"}]},
        updatemenus=[{"buttons": buttons, "active": 1, "x": 0, "y": 1.08, "xanchor": "left"}],
        height=720, margin={"l": 10, "r": 10, "t": 110, "b": 10}, showlegend=False,
    )
    return figure.to_html(include_plotlyjs=True, full_html=True)  # Встраиваем Plotly; подложка грузится из сети.
