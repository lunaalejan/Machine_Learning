import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from LogisticRegression import DATA_PATH


# ============================================================
# 1. CARGAR DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)


# ============================================================
# 2. VARIABLES INDEPENDIENTES
# ============================================================

X = df[
    [
        "edad",
        "visitas_web_mes",
        "tiempo_sitio_min",
        "descuento_usado"
    ]
]


# ============================================================
# 3. VARIABLE OBJETIVO
# ============================================================

y = df["target"]


# ============================================================
# 4. DIVIDIR DATOS 80% ENTRENAMIENTO / 20% PRUEBA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 5. CREAR Y ENTRENAR MODELO
# ============================================================

model = GaussianNB()

model.fit(
    X_train,
    y_train
)


# ============================================================
# 6. PREDICCIONES
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 7. MÉTRICAS DE EVALUACIÓN
# ============================================================

conf_matrix = confusion_matrix(
    y_test,
    y_pred
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# ============================================================
# 8. FUNCIÓN PARA OBTENER LAS MÉTRICAS
# ============================================================

def get_metrics():

    return {
        "confusion_matrix": conf_matrix.tolist(),
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2)
    }


# ============================================================
# 9. FUNCIÓN PARA CLASIFICAR UN NUEVO CLIENTE
# ============================================================

def predict_customer(
    age,
    visits,
    time_on_site,
    discount
):

    new_customer = pd.DataFrame({
        "edad": [age],
        "visitas_web_mes": [visits],
        "tiempo_sitio_min": [time_on_site],
        "descuento_usado": [discount]
    })

    prediction = int(
        model.predict(new_customer)[0]
    )

    probabilities = model.predict_proba(
        new_customer
    )[0]

    purchase_probability = float(
        probabilities[1]
    )

    return prediction, purchase_probability