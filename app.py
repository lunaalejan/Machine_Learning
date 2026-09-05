from flask import Flask, render_template

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

if __name__ == "__main__":
    app.run(debug=True)