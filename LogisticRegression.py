import os
import io
import base64

import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------
# DATASET
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "dataset_regresion_logistica.csv"
)

df = pd.read_csv(DATA_PATH)


# ---------------------------------------------------
# VARIABLES
# ---------------------------------------------------

# Independent variable
X = df[["visitas_web_mes"]]

# Target variable
y = df["target"]


# ---------------------------------------------------
# DATASET INFORMATION
# ---------------------------------------------------

n_records = len(df)

class_0_records = int(
    (df["target"] == 0).sum()
)

class_1_records = int(
    (df["target"] == 1).sum()
)

min_visits = int(
    df["visitas_web_mes"].min()
)

max_visits = int(
    df["visitas_web_mes"].max()
)


# ---------------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

train_records = len(X_train)
test_records = len(X_test)


# ---------------------------------------------------
# LOGISTIC REGRESSION MODEL
# ---------------------------------------------------

logistic_model = LogisticRegression()

logistic_model.fit(
    X_train,
    y_train
)


# ---------------------------------------------------
# TEST PREDICTIONS
# ---------------------------------------------------

y_pred = logistic_model.predict(
    X_test
)


# ---------------------------------------------------
# DATA VISUALIZATION
# ---------------------------------------------------

def create_plot():

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    class_0 = df[
        df["target"] == 0
    ].copy()

    class_1 = df[
        df["target"] == 1
    ].copy()


    # Small visual offsets prevent observations
    # with the same values from completely overlapping.
    class_0_y = []

    for position, _ in enumerate(
        class_0.index
    ):

        offset = (
            (position % 7) - 3
        ) * 0.025

        class_0_y.append(
            0 + offset
        )


    class_1_y = []

    for position, _ in enumerate(
        class_1.index
    ):

        offset = (
            (position % 7) - 3
        ) * 0.025

        class_1_y.append(
            1 + offset
        )


    # CLASS 0
    ax.scatter(
        class_0["visitas_web_mes"],
        class_0_y,
        label="Class 0 - No Purchase",
        alpha=0.65,
        s=45
    )


    # CLASS 1
    ax.scatter(
        class_1["visitas_web_mes"],
        class_1_y,
        label="Class 1 - Purchase",
        alpha=0.65,
        s=45
    )


    # ---------------------------------------------------
    # LOGISTIC PROBABILITY CURVE
    # ---------------------------------------------------

    visit_values = []

    probability_values = []

    current_value = float(
        min_visits
    )

    while current_value <= max_visits:

        visit_values.append(
            current_value
        )

        new_data = pd.DataFrame(
            {
                "visitas_web_mes": [
                    current_value
                ]
            }
        )

        probability = (
            logistic_model.predict_proba(
                new_data
            )[0][1]
        )

        probability_values.append(
            probability
        )

        current_value += 0.1


    ax.plot(
        visit_values,
        probability_values,
        linewidth=2.5,
        label="Purchase Probability"
    )


    # CLASSIFICATION THRESHOLD
    ax.axhline(
        y=0.5,
        linestyle="--",
        linewidth=1.5,
        label="Classification Threshold (0.50)"
    )


    # ---------------------------------------------------
    # GRAPH DESIGN
    # ---------------------------------------------------

    ax.set_title(
        "Customer Classification by Monthly Web Visits",
        fontsize=14,
        pad=15
    )

    ax.set_xlabel(
        "Monthly Web Visits"
    )

    ax.set_ylabel(
        "Class / Probability"
    )

    ax.set_xlim(
        min_visits - 1,
        max_visits + 1
    )

    ax.set_ylim(
        -0.15,
        1.15
    )

    ax.set_yticks(
        [
            0,
            0.5,
            1
        ]
    )

    ax.set_yticklabels(
        [
            "0 - No Purchase",
            "0.50",
            "1 - Purchase"
        ]
    )

    ax.grid(
        alpha=0.20
    )

    ax.legend(
        loc="best"
    )

    plt.tight_layout()


    # ---------------------------------------------------
    # CONVERT GRAPH TO BASE64
    # ---------------------------------------------------

    image = io.BytesIO()

    plt.savefig(
        image,
        format="png",
        dpi=120,
        bbox_inches="tight"
    )

    image.seek(0)

    plot_url = base64.b64encode(
        image.getvalue()
    ).decode("utf-8")

    plt.close(fig)

    return plot_url


plot_image = create_plot()


# ---------------------------------------------------
# NEW CUSTOMER CLASSIFICATION
# ---------------------------------------------------

def predict_purchase(visits):

    visits = float(visits)

    if visits < min_visits or visits > max_visits:

        raise ValueError(
            f"Monthly web visits must be between "
            f"{min_visits} and {max_visits}."
        )

    new_customer = pd.DataFrame(
        {
            "visitas_web_mes": [
                visits
            ]
        }
    )

    prediction = int(
        logistic_model.predict(
            new_customer
        )[0]
    )

    probabilities = (
        logistic_model.predict_proba(
            new_customer
        )[0]
    )

    purchase_probability = float(
        probabilities[1]
    )

    return (
        prediction,
        purchase_probability
    )