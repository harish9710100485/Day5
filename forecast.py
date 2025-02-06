import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import warnings
import csv  # Import the csv module

warnings.filterwarnings("ignore")

# Load Data
file_path = r"C:\Users\haris\Desktop\Day6\Sales_Data_for_Analysis.tsv"

try:
    df = pd.read_csv(file_path, sep="\t")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit()

# Data Cleaning and Preprocessing
df.columns = df.columns.str.strip()

df.rename(columns={"PERIOD": "year", "QTY": "Quantity", "TOTAL PRICE (INR)": "Item Total", "CURRENCY": "Currency"}, inplace=True)

if "year" not in df.columns:
    print("Error: Column 'year' not found in the dataset!")
    exit()

# Convert 'year' column to datetime format
df["year"] = pd.to_datetime(df["year"], errors="coerce", dayfirst=True).dt.year
df.dropna(subset=["year"], inplace=True)

df["year"] = df["year"].astype(int)

# Filter data for INR currency
if "Currency" in df.columns:
    df["Currency"] = df["Currency"].str.strip().str.upper()
    df = df[df["Currency"] == "INR"]

if df.empty:
    print("No INR data available. Exiting.")
    exit()

latest_year = int(df["year"].max())

# Group Data
grouped = df.groupby(["PART NO", "year"])[["Quantity", "Item Total"]].sum().reset_index()

predictions = []

for part_no in grouped["PART NO"].unique():
    part_data = grouped[grouped["PART NO"] == part_no]

    if len(part_data) == 1:
        print(f"Only one year available for {part_no}, copying values for {latest_year + 1}.")
        pred_quantity = part_data["Quantity"].values[0]
        pred_total = part_data["Item Total"].values[0]
    else:
        X = part_data["year"].to_numpy().reshape(-1, 1)
        y_quantity = part_data["Quantity"].to_numpy()
        y_total = part_data["Item Total"].to_numpy()

        # Initialize best models
        best_degree_quantity, best_degree_total = 1, 1
        best_mse_quantity, best_mse_total = float("inf"), float("inf")
        best_model_quantity, best_model_total = None, None
        best_poly_quantity, best_poly_total = None, None

        for degree in range(1, 4):  # Test up to cubic polynomial
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(X)

            # Quantity Model
            model_quantity = LinearRegression().fit(X_poly, y_quantity)
            mse_quantity = mean_squared_error(y_quantity, model_quantity.predict(X_poly))
            if mse_quantity < best_mse_quantity:
                best_mse_quantity = mse_quantity
                best_degree_quantity = degree
                best_model_quantity = model_quantity
                best_poly_quantity = poly

            # Total Model
            model_total = LinearRegression().fit(X_poly, y_total)
            mse_total = mean_squared_error(y_total, model_total.predict(X_poly))
            if mse_total < best_mse_total:
                best_mse_total = mse_total
                best_degree_total = degree
                best_model_total = model_total
                best_poly_total = poly

        next_year = np.array([[latest_year + 1]])

        # Predict Quantity
        next_year_poly_quantity = best_poly_quantity.transform(next_year)
        pred_quantity = best_model_quantity.predict(next_year_poly_quantity)[0]

        # Predict Item Total
        next_year_poly_total = best_poly_total.transform(next_year)
        pred_total = best_model_total.predict(next_year_poly_total)[0]

    predictions.append([part_no, latest_year + 1, pred_quantity, pred_total, "INR"])

# Convert Predictions to DataFrame
predictions_df = pd.DataFrame(predictions, columns=["PART NO", "year", "Predicted Quantity", "Predicted Item Total", "Currency"])

# Save Predictions to TSV, handling commas in data
output_file = r"C:\Users\haris\Desktop\Day6\predictions.tsv"
predictions_df.to_csv(output_file, index=False, sep="\t", quoting=csv.QUOTE_NONNUMERIC)

print(f"Predictions saved in TSV format at: {output_file}")
print(predictions_df)
