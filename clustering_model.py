import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# PATHS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "Online Retail.xlsx"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "clustering_results.csv"
)

CENTROIDS_FILE = os.path.join(
    BASE_DIR,
    "data",
    "clustering_centroids.csv"
)

SUMMARY_FILE = os.path.join(
    BASE_DIR,
    "data",
    "clustering_summary.csv"
)

METRICS_FILE = os.path.join(
    BASE_DIR,
    "data",
    "clustering_metrics.csv"
)

PLOT_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "clustering"
)

os.makedirs(PLOT_FOLDER, exist_ok=True)

# LOAD ORIGINAL DATASET

print("\n======================================")
print("CLUSTERING APPLICATION")
print("======================================")

print("\nLoading Online Retail dataset...")

df = pd.read_excel(INPUT_FILE)

original_records = len(df)

print("Original transaction records:", original_records)

# DATA PREPROCESSING

print("\nPreprocessing data...")

df = df.dropna(subset=["CustomerID"])

df = df.drop_duplicates()

df = df[
    (df["Quantity"] > 0) &
    (df["UnitPrice"] > 0)
].copy()

clean_records = len(df)

print("Valid transaction records:", clean_records)

# FEATURE ENGINEERING

df["TotalSpending"] = (
    df["Quantity"] * df["UnitPrice"]
)

# CUSTOMER-LEVEL DATASET

customers = (
    df.groupby("CustomerID")
    .agg(
        TotalQuantity=("Quantity", "sum"),
        TotalSpending=("TotalSpending", "sum")
    )
    .reset_index()
)

customers["CustomerID"] = (
    customers["CustomerID"].astype(int)
)

customer_count = len(customers)

print("Customer-level records:", customer_count)


if customer_count < 1000:
    raise ValueError(
        "The processed dataset contains fewer than "
        "1,000 customer records."
    )

# VARIABLES USED FOR K-MEANS

X = customers[
    ["TotalQuantity", "TotalSpending"]
].copy()

# FEATURE SCALING

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# K-MEANS MODEL

K = 3

kmeans = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=10
)

customers["Cluster"] = (
    kmeans.fit_predict(X_scaled) + 1
)

# CENTROIDS

centroids_original = scaler.inverse_transform(
    kmeans.cluster_centers_
)

centroids_df = pd.DataFrame(
    centroids_original,
    columns=[
        "TotalQuantity",
        "TotalSpending"
    ]
)

centroids_df.insert(
    0,
    "Cluster",
    range(1, K + 1)
)

centroids_df["TotalQuantity"] = (
    centroids_df["TotalQuantity"].round(2)
)

centroids_df["TotalSpending"] = (
    centroids_df["TotalSpending"].round(2)
)

# SILHOUETTE SCORE

silhouette = silhouette_score(
    X_scaled,
    customers["Cluster"] - 1
)

print(
    "Silhouette Score:",
    round(silhouette, 4)
)

# CLUSTER SUMMARY

summary_df = (
    customers.groupby("Cluster")
    .agg(
        Customers=("CustomerID", "count"),
        AverageQuantity=("TotalQuantity", "mean"),
        AverageSpending=("TotalSpending", "mean")
    )
    .reset_index()
)

summary_df["AverageQuantity"] = (
    summary_df["AverageQuantity"].round(2)
)

summary_df["AverageSpending"] = (
    summary_df["AverageSpending"].round(2)
)

# SCATTER PLOT

plt.figure(figsize=(10, 7))

for cluster in sorted(
    customers["Cluster"].unique()
):

    cluster_data = customers[
        customers["Cluster"] == cluster
    ]

    plt.scatter(
        cluster_data["TotalQuantity"],
        cluster_data["TotalSpending"],
        alpha=0.6,
        label=f"Cluster {cluster}"
    )


plt.scatter(
    centroids_df["TotalQuantity"],
    centroids_df["TotalSpending"],
    marker="X",
    s=250,
    edgecolors="black",
    linewidths=1.5,
    label="Centroids"
)

plt.xlabel("Total Quantity Purchased")
plt.ylabel("Total Spending")

plt.title(
    "Customer Segmentation Using K-Means"
)

plt.legend()

plt.grid(alpha=0.2)

plt.tight_layout()

PLOT_FILE = os.path.join(
    PLOT_FOLDER,
    "clustering_application.png"
)

plt.savefig(
    PLOT_FILE,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

# SAVE RESULTS

customers["TotalSpending"] = (
    customers["TotalSpending"].round(2)
)

customers.to_csv(
    OUTPUT_FILE,
    index=False
)

centroids_df.to_csv(
    CENTROIDS_FILE,
    index=False
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)

# SAVE MODEL METRICS

metrics_df = pd.DataFrame([
    {
        "OriginalTransactions": original_records,
        "CleanTransactions": clean_records,
        "CustomerRecords": customer_count,
        "NumberOfClusters": K,
        "SilhouetteScore": round(
            silhouette,
            4
        )
    }
])

metrics_df.to_csv(
    METRICS_FILE,
    index=False
)



# SHOW RESULTS

print("\n======================================")
print("CLUSTER CENTROIDS")
print("======================================")

print(centroids_df.to_string(index=False))


print("\n======================================")
print("CLUSTER SUMMARY")
print("======================================")

print(summary_df.to_string(index=False))


print("\n======================================")
print("MODEL EVALUATION")
print("======================================")

print(
    "Silhouette Score:",
    round(silhouette, 4)
)


print("\n======================================")
print("FILES CREATED")
print("======================================")

print("- data/clustering_results.csv")
print("- data/clustering_centroids.csv")
print("- data/clustering_summary.csv")
print("- data/clustering_metrics.csv")

print(
    "- static/clustering/"
    "clustering_application.png"
)

print("\nClustering application completed successfully.")