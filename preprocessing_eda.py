# ==========================================
# HOUSE PRICE PREDICTION SYSTEM
# DATA PREPROCESSING + EDA
# ==========================================

# IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# LOAD DATASET
# ==========================================

data = pd.read_csv("dataset/House_Price_Dataset.csv")

print("\n===== DATASET LOADED SUCCESSFULLY =====")


# ==========================================
# SHOW BASIC INFORMATION
# ==========================================

print("\n===== FIRST 5 ROWS =====")
print(data.head())

print("\n===== DATASET SHAPE =====")
print(data.shape)

print("\n===== COLUMN NAMES =====")
print(data.columns)


# ==========================================
# CHECK DUPLICATES
# ==========================================

print("\n===== DUPLICATE ROWS =====")
print(data.duplicated().sum())


# ==========================================
# REMOVE DUPLICATES
# ==========================================

data = data.drop_duplicates()


# ==========================================
# CONVERT DATE COLUMN
# ==========================================

data['date'] = pd.to_datetime(data['date'])

# Extract year from date
data['sale_year'] = data['date'].dt.year


# ==========================================
# DROP UNNECESSARY COLUMNS
# ==========================================

data = data.drop(['id', 'date'], axis=1)

print("\n===== UPDATED COLUMNS =====")
print(data.columns)


# ==========================================
# CREATE VISUALIZATION FOLDER
# ==========================================

if not os.path.exists("visualizations"):
    os.makedirs("visualizations")


# ==========================================
# GRAPH 1 — PRICE DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 5))

sns.histplot(data['price'], kde=True)

plt.title("House Price Distribution")

plt.savefig("visualizations/price_distribution.png")

plt.show()


# ==========================================
# GRAPH 2 — BEDROOMS VS PRICE
# ==========================================

plt.figure(figsize=(10, 5))

sns.scatterplot(x=data['bedrooms'], y=data['price'])

plt.title("Bedrooms vs Price")

plt.savefig("visualizations/bedrooms_vs_price.png")

plt.show()


# ==========================================
# GRAPH 3 — BATHROOMS VS PRICE
# ==========================================

plt.figure(figsize=(10, 5))

sns.scatterplot(x=data['bathrooms'], y=data['price'])

plt.title("Bathrooms vs Price")

plt.savefig("visualizations/bathrooms_vs_price.png")

plt.show()


# ==========================================
# GRAPH 4 — SQFT LIVING VS PRICE
# ==========================================

plt.figure(figsize=(10, 5))

sns.scatterplot(x=data['sqft_living'], y=data['price'])

plt.title("Living Area vs Price")

plt.savefig("visualizations/living_area_vs_price.png")

plt.show()


# ==========================================
# GRAPH 5 — FLOORS VS PRICE
# ==========================================

plt.figure(figsize=(10, 5))

sns.boxplot(x=data['floors'], y=data['price'])

plt.title("Floors vs Price")

plt.savefig("visualizations/floors_vs_price.png")

plt.show()


# ==========================================
# GRAPH 6 — CONDITION VS PRICE
# ==========================================

plt.figure(figsize=(10, 5))

sns.boxplot(x=data['condition'], y=data['price'])

plt.title("Condition vs Price")

plt.savefig("visualizations/condition_vs_price.png")

plt.show()


# ==========================================
# GRAPH 7 — GRADE VS PRICE
# ==========================================

plt.figure(figsize=(10, 5))

sns.boxplot(x=data['grade'], y=data['price'])

plt.title("Grade vs Price")

plt.savefig("visualizations/grade_vs_price.png")

plt.show()


# ==========================================
# GRAPH 8 — CORRELATION HEATMAP
# ==========================================

plt.figure(figsize=(15, 10))

correlation = data.corr()

sns.heatmap(correlation, annot=True, cmap='coolwarm')

plt.title("Correlation Heatmap")

plt.savefig("visualizations/correlation_heatmap.png")

plt.show()


# ==========================================
# SAVE CLEANED DATASET
# ==========================================

data.to_csv("dataset/cleaned_house_data.csv", index=False)

print("\n===== CLEANED DATASET SAVED =====")

print("\n===== EDA COMPLETED SUCCESSFULLY =====")