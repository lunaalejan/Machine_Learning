from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def linear_regression():
    return render_template("LinealRegression.html")

if __name__ == "__main__":
    app.run(debug=True)
    