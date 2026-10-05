import os

from flask import Flask, Response, render_template, request
from matplotlib import pyplot as plt
import pandas as pd


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
    DATA_PATH,
    predict_purchase,
    plot_image as logistic_plot_image,
    n_records as logistic_n_records,
    class_0_records,
    class_1_records,
    train_records,
    test_records,
    min_visits,
    max_visits,
    tn,
    fp,
    fn,
    tp,
    accuracy,
    precision,
    recall,
    f1
)
from NaiveBayes import get_metrics, predict_customer

def read_file(file_path):

    try:
        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:
            return file.read()

    except FileNotFoundError:
        return "File not found."


app = Flask(__name__)


# ---------------------------------------------------
# HOME
# ---------------------------------------------------

@app.route("/")
def home():
    return render_template(
        "home.html"
    )


# ---------------------------------------------------
# TYPES OF MACHINE LEARNING
# ---------------------------------------------------

@app.route("/types-of-machine-learning")
def types_machine_learning():
    return render_template(
        "types_machine_learning.html"
    )


# ---------------------------------------------------
# USE CASES
# ---------------------------------------------------

@app.route("/use-case-1")
def use_case_1():
    return render_template(
        "use_case_1.html"
    )


@app.route("/use-case-2")
def use_case_2():
    return render_template(
        "use_case_2.html"
    )


@app.route("/use-case-3")
def use_case_3():
    return render_template(
        "use_case_3.html"
    )


@app.route("/use-case-4")
def use_case_4():
    return render_template(
        "use_case_4.html"
    )


# ---------------------------------------------------
# LINEAR REGRESSION
# ---------------------------------------------------

@app.route("/LinealRegression")
def Lineal():
    return render_template(
        "LinealRegression.html"
    )


@app.route("/aplication")
def appli():
    return render_template(
        "aplication.html"
    )


@app.route(
    "/model",
    methods=["GET", "POST"]
)
def md():

    storage_input = None

    predicted_price = None

    if request.method == "POST":

        storage_input = request.form.get(
            "storage_gb"
        )

        try:

            storage_value = float(
                storage_input
            )

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


# ---------------------------------------------------
# LOGISTIC REGRESSION - CONCEPTS
# ---------------------------------------------------

@app.route(
    "/logistic-regression/concepts"
)
def logistic_concepts():

    return render_template(
        "logistic_concepts.html"
    )


# ---------------------------------------------------
# LOGISTIC REGRESSION - APPLICATION
# ---------------------------------------------------

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

            (
                prediction_class,
                probability
            ) = predict_purchase(
                visits_value
            )


            purchase_probability = (
                f"{probability * 100:.2f}"
            )


            if prediction_class == 1:

                prediction_label = (
                    "Purchase"
                )

                explanation = (
                    "The model predicts that the "
                    "customer is likely to make "
                    "a purchase."
                )

            else:

                prediction_label = (
                    "No Purchase"
                )

                explanation = (
                    "The model predicts that the "
                    "customer is unlikely to make "
                    "a purchase."
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

        purchase_probability=(
            purchase_probability
        ),

        explanation=explanation,

        error=error
    )


# ---------------------------------------------------
# LOGISTIC REGRESSION - EVALUATION METRICS
# ---------------------------------------------------

@app.route(
    "/logistic-regression/evaluation-metrics"
)
def logistic_metrics():

    return render_template(
        "logistic_metrics.html",

        test_records=test_records,

        tn=int(tn),

        fp=int(fp),

        fn=int(fn),

        tp=int(tp),

        accuracy=f"{accuracy * 100:.2f}",

        precision=f"{precision * 100:.2f}",

        recall=f"{recall * 100:.2f}",

        f1=f"{f1 * 100:.2f}"
    )

#----------------------------------------------------
#NAVI BAYES CONCEPTS
#----------------------------------------------------

@app.route(
    "/naive-bayes/concepts"
)
def naive_bayes_concepts():

    return render_template(
        "naive_bayes_concepts.html"
    )

#---------------------------------------------------
#NAVI BAYES APPLICATION 
#----------------------------------------------------

@app.route(
    "/naive-bayes/application",
    methods=["GET", "POST"]
)
def naive_bayes_application():

    prediction_class = None
    prediction_label = None
    purchase_probability = None
    error = None

    if request.method == "POST":

        try:
            age = float(request.form["edad"])
            visits = float(request.form["visitas_web_mes"])
            time_on_site = float(request.form["tiempo_sitio_min"])
            discount = int(request.form["descuento_usado"])

            prediction_class, probability = predict_customer(
                age,
                visits,
                time_on_site,
                discount
            )

            purchase_probability = f"{probability * 100:.2f}"

            if prediction_class == 1:
                prediction_label = "Purchase"
            else:
                prediction_label = "No Purchase"

        except (ValueError, TypeError) as e:
            error = str(e)

    return render_template(
        "naive_bayes_application.html",
        prediction_class=prediction_class,
        prediction_label=prediction_label,
        purchase_probability=purchase_probability,
        error=error
    )

#------------------------------------------------
#NAIVE BAYES METRICS
#-----------------------------------------------

@app.route("/naive-bayes/metrics")
def naive_bayes_metrics():

    metrics = get_metrics()

    return render_template(
        "naive_bayes_metrics.html",
        metrics=metrics
    )

@app.route("/naive-bayes/visualization")
def naive_bayes_visualization():

    import io
    import pandas as pd
    import matplotlib.pyplot as plt
    from flask import Response

    df = pd.read_csv(DATA_PATH)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Separar los clientes según la clase
    no_purchase = df[df["target"] == 0]
    purchase = df[df["target"] == 1]

    # Graficar clientes que no compraron
    ax.scatter(
        no_purchase["visitas_web_mes"],
        no_purchase["tiempo_sitio_min"],
        label="No Purchase",
        alpha=0.7,
        s=50
    )

    # Graficar clientes que compraron
    ax.scatter(
        purchase["visitas_web_mes"],
        purchase["tiempo_sitio_min"],
        label="Purchase",
        alpha=0.7,
        s=50
    )

    ax.set_title("Website Visits vs. Time on Site")
    ax.set_xlabel("Website Visits per Month")
    ax.set_ylabel("Time on Site (minutes)")

    ax.legend()

    ax.grid(
        True,
        alpha=0.2
    )

    plt.tight_layout()

    image = io.BytesIO()

    plt.savefig(
        image,
        format="png",
        dpi=100
    )

    plt.close(fig)

    image.seek(0)

    return Response(
        image.getvalue(),
        mimetype="image/png"
    )

@app.route('/unsupervised/concepts')
def unsupervised_concepts():
    return render_template('unsupervised_concepts.html')

@app.route('/unsupervised/manual-exercise')
def manual_exercise():

    # --------------------------------------------------
    # FILE PATHS
    # --------------------------------------------------

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    # --------------------------------------------------
    # LOAD THE 100-RECORD DATASET
    # --------------------------------------------------

    records_df = pd.read_csv(
        os.path.join(data_dir, "manual_kmeans_100.csv")
    )

    records = records_df.to_dict(
        orient="records"
    )

    # --------------------------------------------------
    # LOAD THE THREE ITERATIONS
    # --------------------------------------------------

    iteration1_df = pd.read_csv(
        os.path.join(data_dir, "iteration1.csv")
    )

    iteration2_df = pd.read_csv(
        os.path.join(data_dir, "iteration2.csv")
    )

    iteration3_df = pd.read_csv(
        os.path.join(data_dir, "iteration3.csv")
    )

    iteration1 = iteration1_df.to_dict(
        orient="records"
    )

    iteration2 = iteration2_df.to_dict(
        orient="records"
    )

    iteration3 = iteration3_df.to_dict(
        orient="records"
    )

    # --------------------------------------------------
    # LOAD CENTROIDS
    # --------------------------------------------------

    centroids_df = pd.read_csv(
        os.path.join(data_dir, "manual_centroids.csv")
    )

    def get_centroids(stage):

        selected = centroids_df[
            centroids_df["Stage"] == stage
        ]

        return selected[
            ["TotalQuantity", "TotalSpending"]
        ].values.tolist()

    initial_centroids = get_centroids("Initial")
    centroids1 = get_centroids("Iteration 1")
    centroids2 = get_centroids("Iteration 2")
    centroids3 = get_centroids("Iteration 3")

    # --------------------------------------------------
    # LOAD VARIATION RESULTS
    # --------------------------------------------------

    variance_df = pd.read_csv(
        os.path.join(data_dir, "manual_variance.csv")
    )

    variance_data = variance_df.to_dict(
        orient="records"
    )

    # --------------------------------------------------
    # INTERPRETATION OF WITHIN-CLUSTER VARIATION
    # --------------------------------------------------

    variance_interpretation = (
        "The total within-cluster sum of squares (WCSS) decreased "
        "from 30,840,040.46 in Iteration 1 to 27,682,280.33 in "
        "Iteration 2 and 25,172,487.73 in Iteration 3. This "
        "progressive reduction indicates that the observations "
        "became more compact around their assigned centroids during "
        "the three manual iterations."
    )

    # --------------------------------------------------
    # FINAL CLUSTER INTERPRETATION
    # --------------------------------------------------

    cluster1_interpretation = (
        "Cluster 1 has a final centroid of approximately "
        "(164.61, 337.29). Compared with the other two clusters, "
        "these customers show lower total quantities purchased and "
        "lower total spending within this 100-customer sample."
    )

    cluster2_interpretation = (
        "Cluster 2 has a final centroid of approximately "
        "(636.15, 1051.43). These customers show intermediate "
        "purchasing behavior, with higher quantity and spending "
        "than Cluster 1 but lower values than Cluster 3."
    )

    cluster3_interpretation = (
        "Cluster 3 has a final centroid of approximately "
        "(1495.60, 2934.75). Compared with the other clusters, "
        "these customers show the highest total quantities "
        "purchased and the highest total spending within the "
        "manual sample."
    )

    # --------------------------------------------------
    # FINAL CONCLUSION
    # --------------------------------------------------

    manual_conclusion = (
        "The three-iteration manual K-Means simulation demonstrates "
        "how customer observations are progressively reorganized "
        "according to their Euclidean distance from the centroids. "
        "After each assignment step, new centroids were calculated "
        "using the mean values of the observations in each cluster. "
        "The reduction in total WCSS across the three iterations "
        "shows an improvement in cluster compactness. The final "
        "centroids also reveal three distinct levels of purchasing "
        "behavior in terms of total quantity and total spending "
        "within the selected sample."
    )

    # PLOT PATHS
    
    initial_plot = "kmeans_manual/initial_plot.png"
    iteration1_plot = "kmeans_manual/iteration1.png"
    iteration2_plot = "kmeans_manual/iteration2.png"
    iteration3_plot = "kmeans_manual/iteration3.png"

    # --------------------------------------------------
    # SEND EVERYTHING TO THE HTML TEMPLATE
    # --------------------------------------------------

    return render_template(
        "manual_exercise.html",

        records=records,

        initial_centroids=initial_centroids,
        initial_plot=initial_plot,

        iteration1=iteration1,
        centroids1=centroids1,
        iteration1_plot=iteration1_plot,

        iteration2=iteration2,
        centroids2=centroids2,
        iteration2_plot=iteration2_plot,

        iteration3=iteration3,
        centroids3=centroids3,
        iteration3_plot=iteration3_plot,

        variance_data=variance_data,
        variance_interpretation=variance_interpretation,

        cluster1_interpretation=cluster1_interpretation,
        cluster2_interpretation=cluster2_interpretation,
        cluster3_interpretation=cluster3_interpretation,

        manual_conclusion=manual_conclusion
    )

@app.route('/unsupervised/clustering-application')
def clustering_application():

    import os
    import pandas as pd


    # ---------------------------------------------
    # PATHS
    # ---------------------------------------------

    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    data_dir = os.path.join(
        base_dir,
        "data"
    )


    # ---------------------------------------------
    # LOAD RESULTS
    # ---------------------------------------------

    results_df = pd.read_csv(
        os.path.join(
            data_dir,
            "clustering_results.csv"
        )
    )


    centroids_df = pd.read_csv(
        os.path.join(
            data_dir,
            "clustering_centroids.csv"
        )
    )


    summary_df = pd.read_csv(
        os.path.join(
            data_dir,
            "clustering_summary.csv"
        )
    )


    metrics_df = pd.read_csv(
        os.path.join(
            data_dir,
            "clustering_metrics.csv"
        )
    )


    # Convert tables for HTML

    records = results_df.to_dict(
        orient="records"
    )


    centroids = centroids_df.to_dict(
        orient="records"
    )


    summary = summary_df.to_dict(
        orient="records"
    )


    silhouette_score = round(
        metrics_df[
            "SilhouetteScore"
        ][0],
        4
    )


    return render_template(

        "clustering_application.html",

        records=records,

        centroids=centroids,

        summary=summary,

        silhouette_score=silhouette_score,

        plot="clustering/clustering_application.png"

    )

@app.route("/reinforcement-learning")
def reinforcement_learning():

    base_path = os.path.join(
        app.static_folder,
        "reinforcement_learning"
    )

    training_results = read_file(
        os.path.join(base_path, "training_results.txt")
    )

    q_values = read_file(
        os.path.join(base_path, "q_values.txt")
    )

    agent_path = read_file(
        os.path.join(base_path, "agent_path.txt")
    )

    concepts = read_file(
        os.path.join(base_path, "concepts.txt")
    )

    return render_template(
        "reinforcement_learning.html",
        training_results=training_results,
        q_values=q_values,
        agent_path=agent_path,
        concepts=concepts
    )

# ---------------------------------------------------
# RUN APPLICATION
# ---------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )