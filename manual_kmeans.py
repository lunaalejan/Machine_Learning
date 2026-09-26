import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "manual_kmeans_100.csv"
)

PLOT_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "kmeans_manual"
)

os.makedirs(PLOT_FOLDER, exist_ok=True)


# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(DATA_FILE)

X = df[
    ["TotalQuantity", "TotalSpending"]
].to_numpy(dtype=float)

print("\n======================================")
print("MANUAL K-MEANS SIMULATION")
print("======================================")

print("Number of records:", len(df))


# ==========================================================
# SELECT THREE INITIAL CENTROIDS
# ==========================================================
# We select representative observations according to
# TotalSpending: low, middle and high positions.
#
# These are REAL observations from the 100-record sample.

ordered = df.sort_values(
    "TotalSpending"
).reset_index(drop=True)

selected_positions = [10, 49, 89]

initial_centroids = ordered.loc[
    selected_positions,
    ["TotalQuantity", "TotalSpending"]
].to_numpy(dtype=float)


print("\nINITIAL CENTROIDS")

for i, centroid in enumerate(initial_centroids, start=1):
    print(
        f"C{i}: "
        f"TotalQuantity={centroid[0]:.2f}, "
        f"TotalSpending={centroid[1]:.2f}"
    )


# ==========================================================
# EUCLIDEAN DISTANCE
# ==========================================================

def euclidean_distance(point, centroid):

    return np.sqrt(
        (point[0] - centroid[0]) ** 2 +
        (point[1] - centroid[1]) ** 2
    )


# ==========================================================
# RUN ONE MANUAL ITERATION
# ==========================================================

def run_iteration(dataframe, centroids):

    results = []

    cluster_points = {
        1: [],
        2: [],
        3: []
    }

    for _, row in dataframe.iterrows():

        point = np.array([
            float(row["TotalQuantity"]),
            float(row["TotalSpending"])
        ])

        distances = [
            euclidean_distance(point, centroid)
            for centroid in centroids
        ]

        assigned_cluster = (
            int(np.argmin(distances)) + 1
        )

        cluster_points[
            assigned_cluster
        ].append(point)

        results.append({
            "CustomerID": int(row["CustomerID"]),
            "X": round(point[0], 2),
            "Y": round(point[1], 2),
            "DistanceC1": round(distances[0], 2),
            "DistanceC2": round(distances[1], 2),
            "DistanceC3": round(distances[2], 2),
            "Cluster": assigned_cluster
        })

    # ------------------------------------------
    # UPDATE CENTROIDS
    # ------------------------------------------

    new_centroids = []

    for cluster_number in [1, 2, 3]:

        points = cluster_points[cluster_number]

        if len(points) > 0:

            points_array = np.array(points)

            new_centroid = points_array.mean(
                axis=0
            )

        else:
            # Safety measure in case a cluster
            # receives no observations.
            new_centroid = centroids[
                cluster_number - 1
            ]

        new_centroids.append(new_centroid)

    return (
        results,
        np.array(new_centroids)
    )


# ==========================================================
# WITHIN-CLUSTER VARIATION
# ==========================================================
# Here variation is measured as the mean squared Euclidean
# distance between observations and their UPDATED centroid.
# We also calculate total WCSS (sum of squared distances).

def calculate_variation(results, centroids):

    cluster_squared_distances = {
        1: [],
        2: [],
        3: []
    }

    for row in results:

        cluster = row["Cluster"]

        point = np.array([
            row["X"],
            row["Y"]
        ], dtype=float)

        centroid = centroids[
            cluster - 1
        ]

        squared_distance = np.sum(
            (point - centroid) ** 2
        )

        cluster_squared_distances[
            cluster
        ].append(squared_distance)

    cluster_variances = {}

    total_wcss = 0

    for cluster in [1, 2, 3]:

        values = cluster_squared_distances[
            cluster
        ]

        if values:

            cluster_variances[cluster] = (
                float(np.mean(values))
            )

            total_wcss += float(
                np.sum(values)
            )

        else:

            cluster_variances[cluster] = 0.0

    return cluster_variances, total_wcss


# ==========================================================
# CREATE SCATTER PLOT
# ==========================================================

def create_plot(
    dataframe,
    results,
    centroids,
    filename,
    title
):

    plt.figure(figsize=(9, 6))

    if results is None:

        plt.scatter(
            dataframe["TotalQuantity"],
            dataframe["TotalSpending"],
            alpha=0.7,
            label="Customers"
        )

    else:

        result_df = pd.DataFrame(results)

        for cluster in [1, 2, 3]:

            cluster_data = result_df[
                result_df["Cluster"] == cluster
            ]

            plt.scatter(
                cluster_data["X"],
                cluster_data["Y"],
                alpha=0.7,
                label=f"Cluster {cluster}"
            )

    # Centroids
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="X",
        s=220,
        edgecolors="black",
        linewidths=1.5,
        label="Centroids"
    )

    plt.xlabel("Total Quantity Purchased")
    plt.ylabel("Total Spending")
    plt.title(title)

    plt.legend()
    plt.grid(alpha=0.2)

    plt.tight_layout()

    full_path = os.path.join(
        PLOT_FOLDER,
        filename
    )

    plt.savefig(
        full_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


# ==========================================================
# INITIAL PLOT
# ==========================================================

create_plot(
    df,
    None,
    initial_centroids,
    "initial_plot.png",
    "Initial Dataset and Centroids"
)


# ==========================================================
# ITERATION 1
# ==========================================================

print("\n======================================")
print("ITERATION 1")
print("======================================")

iteration1, centroids1 = run_iteration(
    df,
    initial_centroids
)

variance1, wcss1 = calculate_variation(
    iteration1,
    centroids1
)

print("\nUpdated centroids:")

for i, centroid in enumerate(
    centroids1,
    start=1
):

    print(
        f"C{i}: "
        f"({centroid[0]:.2f}, "
        f"{centroid[1]:.2f})"
    )

print("Total WCSS:", round(wcss1, 2))


create_plot(
    df,
    iteration1,
    centroids1,
    "iteration1.png",
    "K-Means - Iteration 1"
)


# ==========================================================
# ITERATION 2
# ==========================================================

print("\n======================================")
print("ITERATION 2")
print("======================================")

iteration2, centroids2 = run_iteration(
    df,
    centroids1
)

variance2, wcss2 = calculate_variation(
    iteration2,
    centroids2
)

print("\nUpdated centroids:")

for i, centroid in enumerate(
    centroids2,
    start=1
):

    print(
        f"C{i}: "
        f"({centroid[0]:.2f}, "
        f"{centroid[1]:.2f})"
    )

print("Total WCSS:", round(wcss2, 2))


create_plot(
    df,
    iteration2,
    centroids2,
    "iteration2.png",
    "K-Means - Iteration 2"
)


# ==========================================================
# ITERATION 3
# ==========================================================

print("\n======================================")
print("ITERATION 3")
print("======================================")

iteration3, centroids3 = run_iteration(
    df,
    centroids2
)

variance3, wcss3 = calculate_variation(
    iteration3,
    centroids3
)

print("\nFinal centroids:")

for i, centroid in enumerate(
    centroids3,
    start=1
):

    print(
        f"C{i}: "
        f"({centroid[0]:.2f}, "
        f"{centroid[1]:.2f})"
    )

print("Total WCSS:", round(wcss3, 2))


create_plot(
    df,
    iteration3,
    centroids3,
    "iteration3.png",
    "K-Means - Iteration 3"
)


# ==========================================================
# SAVE ITERATION TABLES
# ==========================================================

pd.DataFrame(
    iteration1
).to_csv(
    os.path.join(
        BASE_DIR,
        "data",
        "iteration1.csv"
    ),
    index=False
)


pd.DataFrame(
    iteration2
).to_csv(
    os.path.join(
        BASE_DIR,
        "data",
        "iteration2.csv"
    ),
    index=False
)


pd.DataFrame(
    iteration3
).to_csv(
    os.path.join(
        BASE_DIR,
        "data",
        "iteration3.csv"
    ),
    index=False
)


# ==========================================================
# SAVE CENTROIDS
# ==========================================================

centroid_rows = []

all_centroids = [
    ("Initial", initial_centroids),
    ("Iteration 1", centroids1),
    ("Iteration 2", centroids2),
    ("Iteration 3", centroids3)
]

for stage, centroid_set in all_centroids:

    for index, centroid in enumerate(
        centroid_set,
        start=1
    ):

        centroid_rows.append({
            "Stage": stage,
            "Centroid": f"C{index}",
            "TotalQuantity": round(
                centroid[0],
                2
            ),
            "TotalSpending": round(
                centroid[1],
                2
            )
        })


pd.DataFrame(
    centroid_rows
).to_csv(
    os.path.join(
        BASE_DIR,
        "data",
        "manual_centroids.csv"
    ),
    index=False
)


# ==========================================================
# SAVE VARIANCE COMPARISON
# ==========================================================

variance_table = pd.DataFrame([
    {
        "Iteration": "Iteration 1",
        "Cluster1": round(
            variance1[1],
            2
        ),
        "Cluster2": round(
            variance1[2],
            2
        ),
        "Cluster3": round(
            variance1[3],
            2
        ),
        "Total": round(
            wcss1,
            2
        )
    },
    {
        "Iteration": "Iteration 2",
        "Cluster1": round(
            variance2[1],
            2
        ),
        "Cluster2": round(
            variance2[2],
            2
        ),
        "Cluster3": round(
            variance2[3],
            2
        ),
        "Total": round(
            wcss2,
            2
        )
    },
    {
        "Iteration": "Iteration 3",
        "Cluster1": round(
            variance3[1],
            2
        ),
        "Cluster2": round(
            variance3[2],
            2
        ),
        "Cluster3": round(
            variance3[3],
            2
        ),
        "Total": round(
            wcss3,
            2
        )
    }
])


variance_table.to_csv(
    os.path.join(
        BASE_DIR,
        "data",
        "manual_variance.csv"
    ),
    index=False
)


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n======================================")
print("SIMULATION COMPLETED")
print("======================================")

print("\nWCSS comparison:")

print(
    "Iteration 1:",
    round(wcss1, 2)
)

print(
    "Iteration 2:",
    round(wcss2, 2)
)

print(
    "Iteration 3:",
    round(wcss3, 2)
)

print("\nFiles created:")

print("- data/iteration1.csv")
print("- data/iteration2.csv")
print("- data/iteration3.csv")
print("- data/manual_centroids.csv")
print("- data/manual_variance.csv")

print("\nPlots created:")

print("- static/kmeans_manual/initial_plot.png")
print("- static/kmeans_manual/iteration1.png")
print("- static/kmeans_manual/iteration2.png")
print("- static/kmeans_manual/iteration3.png")

print("\nManual K-Means simulation completed successfully.")