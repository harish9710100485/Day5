# Sales Data Analysis & Forecasting

## Overview
This project analyzes sales data and predicts future sales using **Polynomial Regression**. It reads sales data from a TSV file, cleans and processes the data, applies machine learning models, and forecasts sales for the next year.

## Features
- **Data Preprocessing**: Handles missing values, renames columns, and filters relevant data.
- **Currency Filtering**: Filters sales data only for INR transactions.
- **Polynomial Regression**: Fits models to predict sales trends.
- **Automated Degree Selection**: Chooses the best polynomial degree (up to 3) using Mean Squared Error.
- **Forecasting**: Predicts sales for the next year.
- **TSV Output**: Saves predictions in TSV format.

## Technology Stack
- **Python** for scripting
- **Pandas** for data handling
- **NumPy** for numerical operations
- **Scikit-Learn** for machine learning
- **Matplotlib** (optional) for visualization
- **CSV Module** for data storage

## File Structure
```
project-folder/
│── sales_forecast.py      # Main script for forecasting
│── Sales_Data_for_Analysis.tsv  # Input data file
│── predictions.tsv        # Output file with predictions
```

## Installation & Usage
1. Clone or download the repository.
2. Install required dependencies:
   ```bash
   pip install pandas numpy scikit-learn
   ```
3. Update the `file_path` variable in `sales_forecast.py` to point to your dataset.
4. Run the script:
   ```bash
   python sales_forecast.py
   ```
5. The predictions will be saved as `predictions.tsv`.

## Data Processing
- **Column Cleaning**: Trims and renames key columns.
- **Date Conversion**: Converts 'year' column to integer values.
- **Missing Data Handling**: Drops invalid entries.
- **Grouping**: Aggregates sales by **PART NO** and **year**.

## Prediction Methodology
- Uses **Polynomial Regression** to fit the best model for sales trends.
- Tests polynomial degrees from **1 to 3** and selects the best using **Mean Squared Error**.
- If only one year's data is available, it copies values forward.
- Predicts sales quantity and total price for the next year.

## Output Format
- The results are stored in `predictions.tsv` with the following columns:
  - **PART NO**: Product identifier
  - **year**: Predicted year
  - **Predicted Quantity**: Forecasted sales quantity
  - **Predicted Item Total**: Forecasted total sales amount
  - **Currency**: INR (filtered data)

## Example Output
```
PART NO  | year  | Predicted Quantity | Predicted Item Total | Currency
---------------------------------------------------------------------
12345    | 2026  | 500                | 150000               | INR
67890    | 2026  | 1200               | 300000               | INR
```

## Conclusion
This project provides an automated pipeline to process sales data and predict future trends using **machine learning** techniques. It helps businesses forecast inventory and sales, ensuring better decision-making.

