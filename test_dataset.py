# ===============================
# HOUSE PRICE PREDICTION PROJECT
# STEP 1: LOAD DATASET
# ===============================

# Import pandas
import pandas as pd


# ===============================
# LOAD DATASET
# ===============================

# Replace filename with your actual CSV filename
data = pd.read_csv("dataset/House_Price_Dataset.csv")


# ===============================
# SHOW FIRST 5 ROWS
# ===============================

print("\n===== FIRST 5 ROWS =====")

print(data.head())


# ===============================
# SHOW COLUMN NAMES
# ===============================

print("\n===== COLUMN NAMES =====")

print(data.columns)


# ===============================
# SHOW DATASET SHAPE
# ===============================

print("\n===== DATASET SHAPE =====")

print(data.shape)


# ===============================
# SHOW DATASET INFORMATION
# ===============================

print("\n===== DATASET INFO =====")

print(data.info())


# ===============================
# CHECK MISSING VALUES
# ===============================

print("\n===== MISSING VALUES =====")

print(data.isnull().sum())