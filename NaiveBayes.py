import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

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
# 4. DIVIDIR LOS DATOS EN ENTRENAMIENTO Y PRUEBA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 5. CREAR Y ENTRENAR EL MODELO NAIVE BAYES
# ============================================================

model = GaussianNB()

model.fit(
    X_train,
    y_train
)


# ============================================================
# 6. REALIZAR PREDICCIONES SOBRE LOS DATOS DE PRUEBA
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 7. FUNCIÓN PARA CLASIFICAR UN NUEVO CLIENTE
# ============================================================

def predict_customer(
    age,
    visits,
    time_on_site,
    discount
):

    # Crear DataFrame con los datos del nuevo cliente
    new_customer = pd.DataFrame({
        "edad": [age],
        "visitas_web_mes": [visits],
        "tiempo_sitio_min": [time_on_site],
        "descuento_usado": [discount]
    })


    # Realizar predicción
    prediction = int(
        model.predict(new_customer)[0]
    )


    # Obtener probabilidades de cada clase
    probabilities = model.predict_proba(
        new_customer
    )[0]


    # Probabilidad de la clase 1
    purchase_probability = float(
        probabilities[1]
    )


    # Retornar clase y probabilidad
    return prediction, purchase_probability