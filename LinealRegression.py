import io
import base64
from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. CARGAR DATASET
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "computer_storage_price_dataset.csv"

df = pd.read_csv(DATA_PATH)


# ============================================================
# 2. ENTRENAR MODELO
# ============================================================

X = df[["Storage_GB"]]
y = df["Price_COP"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)


# ============================================================
# 3. HACER PREDICCIONES PARA EVALUAR EL MODELO
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 4. MÉTRICAS DEL MODELO
# ============================================================

intercept = model.intercept_
coef = model.coef_[0]

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

n_records = len(df)


# ============================================================
# 5. FUNCIÓN PARA PREDECIR EL PRECIO
# ============================================================

def calculatePrice(storage_gb):

    storage_data = pd.DataFrame({
        "Storage_GB": [storage_gb]
    })

    prediction = model.predict(storage_data)

    return float(prediction[0])


# ============================================================
# 6. CREAR GRÁFICA
# ============================================================

def build_plot():

    fig, ax = plt.subplots(
        figsize=(8, 5.5),
        dpi=150
    )

    BG_2 = "#1D2227"
    INK = "#EDEAE0"
    MUTED = "#A5ACB3"
    OCHRE = "#D99B3C"
    TEAL = "#4FA393"
    LINE = "#333B42"

    fig.patch.set_facecolor(BG_2)
    ax.set_facecolor(BG_2)

    # Datos reales
    ax.scatter(
        df["Storage_GB"],
        df["Price_COP"],
        color=TEAL,
        alpha=0.45,
        s=22,
        edgecolors="none",
        label="Actual data"
    )

    # Valores para la línea de regresión
    x_range = np.linspace(
        df["Storage_GB"].min(),
        df["Storage_GB"].max(),
        200
    )

    x_range_df = pd.DataFrame({
        "Storage_GB": x_range
    })

    y_range = model.predict(x_range_df)

    # Línea de regresión
    ax.plot(
        x_range,
        y_range,
        color=OCHRE,
        linewidth=2.5,
        label="Regression line"
    )

    # Título
    ax.set_title(
        "Computer Price vs. Storage Capacity (Simple Linear Regression)",
        color=INK,
        fontsize=13,
        pad=14
    )

    # Ejes
    ax.set_xlabel(
        "Storage capacity (GB)",
        color=INK,
        fontsize=11
    )

    ax.set_ylabel(
        "Price (COP)",
        color=INK,
        fontsize=11
    )

    # Configuración visual
    ax.tick_params(colors=MUTED)

    for spine in ax.spines.values():
        spine.set_color(LINE)

    ax.grid(
        True,
        color=LINE,
        linewidth=0.6,
        alpha=0.5
    )

    ax.legend(
        facecolor=BG_2,
        edgecolor=LINE,
        labelcolor=INK,
        fontsize=9
    )

    fig.tight_layout()

    # Convertir gráfica a imagen
    buffer = io.BytesIO()

    fig.savefig(
        buffer,
        format="png",
        facecolor=fig.get_facecolor()
    )

    plt.close(fig)

    buffer.seek(0)

    image_base64 = base64.b64encode(
        buffer.read()
    ).decode("utf-8")

    return image_base64

plot_image = build_plot()
