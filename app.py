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

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

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

if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )