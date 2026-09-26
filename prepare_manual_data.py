import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

input_file = os.path.join(
    BASE_DIR,
    "data",
    "Online Retail.xlsx"
)

output_file = os.path.join(
    BASE_DIR,
    "data",
    "manual_kmeans_100.csv"
)

print("Loading Online Retail dataset...")

df = pd.read_excel(input_file)

print("Original records:", len(df))

df = df.dropna(subset=["CustomerID"])
df = df.drop_duplicates()
df = df[
    (df["Quantity"] > 0) &
    (df["UnitPrice"] > 0)
].copy()

print("Records after cleaning:", len(df))

df["TotalSpending"] = (
    df["Quantity"] * df["UnitPrice"]
)

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

print("Customers after grouping:", len(customers))

quantity_limit = customers[
    "TotalQuantity"
].quantile(0.95)

spending_limit = customers[
    "TotalSpending"
].quantile(0.95)

manual_pool = customers[
    (customers["TotalQuantity"] <= quantity_limit) &
    (customers["TotalSpending"] <= spending_limit)
].copy()

print(
    "Customers available for manual exercise:",
    len(manual_pool)
)

manual_data = manual_pool.sample(
    n=100,
    random_state=42
).copy()

manual_data = manual_data.sort_values(
    by="CustomerID"
).reset_index(drop=True)

manual_data["TotalSpending"] = (
    manual_data["TotalSpending"].round(2)
)

manual_data.to_csv(
    output_file,
    index=False
)

print("\n--------------------------------")
print("MANUAL K-MEANS DATASET CREATED")
print("--------------------------------")

print("\nNumber of records:")
print(len(manual_data))

print("\nVariables:")
print(manual_data.columns.tolist())

print("\nFirst 10 records:")
print(manual_data.head(10))

print("\nFile saved in:")
print(output_file)