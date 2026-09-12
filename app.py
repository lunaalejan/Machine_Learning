from flask import Flask, render_template, request

from LinealRegression import (
    calculatePrice,
    plot_image,
    n_records,
    intercept,
    coef,
    mae,
    rmse,
    r2
)

from LogisticRegression import (
    predict_purchase,
    plot_image as logistic_plot_image,
    n_records as logistic_n_records,
    class_0_records,
    class_1_records,
    train_records,
    test_records,
    min_visits,
    max_visits
)


app = Flask(__name__)


# HOME
@app.route("/")
def home():
    return render_template("home.html")


# TYPES OF MACHINE LEARNING
@app.route("/types-of-machine-learning")
def types_machine_learning():
    return render_template("types_machine_learning.html")


# USE CASES
@app.route("/use-case-1")
def use_case_1():
    return render_template("use_case_1.html")


@app.route("/use-case-2")
def use_case_2():
    return render_template("use_case_2.html")


@app.route("/use-case-3")
def use_case_3():
    return render_template("use_case_3.html")


@app.route("/use-case-4")
def use_case_4():
    return render_template("use_case_4.html")


# LINEAR REGRESSION
@app.route("/LinealRegression")
def Lineal():
    return render_template("LinealRegression.html")


@app.route("/aplication")
def appli():
    return render_template("aplication.html")


@app.route("/model", methods=["GET", "POST"])
def md():

    storage_input = None
    predicted_price = None

    if request.method == "POST":

        storage_input = request.form.get("storage_gb")

        try:
            storage_value = float(storage_input)

            predicted_price = calculatePrice(
                storage_value
            )

        except (ValueError, TypeError):
            predicted_price = None

    return render_template(
        "model.html",

        plot_image=plot_image,

        n_records=n_records,

        intercept=f"{intercept:,.2f}",

        coef=f"{coef:,.4f}",

        mae=f"{mae:,.2f}",

        rmse=f"{rmse:,.2f}",

        r2=f"{r2:.4f}",

        storage_input=storage_input,

        predicted_price=(
            f"{predicted_price:,.0f}"
            if predicted_price is not None
            else None
        )
    )


# LOGISTIC REGRESSION - CONCEPTS
@app.route("/logistic-regression/concepts")
def logistic_concepts():

    return render_template(
        "logistic_concepts.html"
    )


# LOGISTIC REGRESSION - APPLICATION
@app.route(
    "/logistic-regression/application",
    methods=["GET", "POST"]
)
def logistic_application():

    visits_input = None

    prediction_class = None
    prediction_label = None
    purchase_probability = None

    explanation = None
    error = None

    if request.method == "POST":

        visits_input = request.form.get(
            "visitas_web_mes"
        )

        try:

            visits_value = float(
                visits_input
            )

            prediction_class, probability = predict_purchase(
                visits_value
            )

            purchase_probability = (
                f"{probability * 100:.2f}"
            )

            if prediction_class == 1:

                prediction_label = "Purchase"

                explanation = (
                    "The model predicts that the customer "
                    "is likely to make a purchase."
                )

            else:

                prediction_label = "No Purchase"

                explanation = (
                    "The model predicts that the customer "
                    "is unlikely to make a purchase."
                )

        except (ValueError, TypeError) as e:

            error = str(e)

    return render_template(
        "logistic_application.html",

        n_records=logistic_n_records,

        class_0_records=class_0_records,

        class_1_records=class_1_records,

        train_records=train_records,

        test_records=test_records,

        min_visits=min_visits,

        max_visits=max_visits,

        plot_image=logistic_plot_image,

        visits_input=visits_input,

        prediction_class=prediction_class,

        prediction_label=prediction_label,

        purchase_probability=purchase_probability,

        explanation=explanation,

        error=error
    )


if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )