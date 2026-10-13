import pandas as pd

df = pd.read_csv(r"C:\Users\asus\OneDrive\Documents\Desktop\Sales_Capstone_Project\_Dataset\sales_data.csv")
print("Dataset loaded successfully!")
print("Rows and Columns:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())
#Basic Information
print("\nDataset Information:")
print(df.info())

#Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

#Check duplicate Rows
print("\nDuplicate Rows:", df.duplicated().sum())

#Check Missing values
print("\nMissing Values:")
print(df.isnull().sum())
# ---------------- DATA CLEANING ----------------
#Filling Missing Discount Band values
df["Discount Band"] = df["Discount Band"].fillna("Unknown")

#Convert Date column to determine format
df["Date"] = pd.to_datetime(df["Date"])

# Check missing values after cleaning
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

#check data types
print("\nData Types:")
print(df.dtypes)

#Final dataset shape
print("\nFinal Dataset Shape:", df.shape)
# ---------------- EDA ANALYSIS ----------------

# Total Sales
total_sales = df[" Sales"].sum()

# Total Profit
total_profit = df["Profit"].sum()

# Total Units Sold
total_units = df["Units Sold"].sum()

print("\n===== KEY BUSINESS METRICS =====")
print("Total Sales:", round(total_sales, 2))
print("Total Profit:", round(total_profit, 2))
print("Total Units Sold:", round(total_units, 2))


# Sales by Country
country_sales = df.groupby("Country")[" Sales"].sum().sort_values(ascending=False)

print("\n===== SALES BY COUNTRY =====")
print(country_sales)


# Sales by Product
product_sales = df.groupby("Product")[" Sales"].sum().sort_values(ascending=False)

print("\n===== SALES BY PRODUCT =====")
print(product_sales)


# Profit by Product
product_profit = df.groupby("Product")["Profit"].sum().sort_values(ascending=False)

print("\n===== PROFIT BY PRODUCT =====")
print(product_profit)


# Sales by Segment
segment_sales = df.groupby("Segment")[" Sales"].sum().sort_values(ascending=False)

print("\n===== SALES BY SEGMENT =====")
print(segment_sales)


# Year-wise Sales
year_sales = df.groupby("Year")[" Sales"].sum()

print("\n===== YEAR-WISE SALES =====")
print(year_sales)


# Year-wise Profit
year_profit = df.groupby("Year")["Profit"].sum()

print("\n===== YEAR-WISE PROFIT =====")
print(year_profit)
# ---------------- EDA VISUALIZATIONS ----------------

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Sales by Country
plt.figure(figsize=(10, 6))
country_sales.plot(kind="bar")
plt.title("Sales by Country")
plt.xlabel("Country")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 2. Sales by Product
plt.figure(figsize=(10, 6))
product_sales.plot(kind="bar")
plt.title("Sales by Product")
plt.xlabel("Product")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 3. Profit by Product
plt.figure(figsize=(10, 6))
product_profit.plot(kind="bar")
plt.title("Profit by Product")
plt.xlabel("Product")
plt.ylabel("Profit")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 4. Sales by Segment
plt.figure(figsize=(10, 6))
segment_sales.plot(kind="bar")
plt.title("Sales by Segment")
plt.xlabel("Segment")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 5. Year-wise Sales
plt.figure(figsize=(8, 5))
year_sales.plot(kind="bar")
plt.title("Year-wise Sales")
plt.xlabel("Year")
plt.ylabel("Sales")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# 6. Year-wise Profit
plt.figure(figsize=(8, 5))
year_profit.plot(kind="bar")
plt.title("Year-wise Profit")
plt.xlabel("Year")
plt.ylabel("Profit")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
# ---------------- SALES PREDICTION MODEL ----------------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Create a clean Sales column
df["Sales"] = df[" Sales"]

# Create useful date features
df["Month"] = df["Date"].dt.month

# Select features for prediction
features = [
    "Segment",
    "Country",
    "Product",
    "Discount Band",
    "Units Sold",
    "Manufacturing Price",
    "Sale Price",
    "Month",
    "Year"
]

X = df[features]
y = df["Sales"]

# Convert categorical columns into numerical columns
X = pd.get_dummies(X, drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n===== REGRESSION MODEL RESULTS =====")
print("Mean Absolute Error (MAE):", round(mae, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))
print("R² Score:", round(r2, 4))
# ---------------- ACTUAL VS PREDICTED SALES ----------------

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

plt.tight_layout()
plt.show()