# ==========================================
# HOUSE PRICE PREDICTION SYSTEM
# MACHINE LEARNING MODEL TRAINING
# ==========================================

# IMPORT LIBRARIES
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# LOAD CLEANED DATASET
# ==========================================

data = pd.read_csv("dataset/cleaned_house_data.csv")

print("\n===== CLEANED DATASET LOADED =====")


# ==========================================
# FEATURES & TARGET
# ==========================================

X = data.drop("price", axis=1)

y = data["price"]


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n===== TRAIN TEST SPLIT COMPLETED =====")


# ==========================================
# LINEAR REGRESSION MODEL
# ==========================================

print("\n===== TRAINING LINEAR REGRESSION =====")

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

print("Linear Regression Training Completed!")


# ==========================================
# RANDOM FOREST MODEL
# ==========================================

print("\n===== TRAINING RANDOM FOREST =====")

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

print("Random Forest Training Completed!")


# ==========================================
# EVALUATION FUNCTION
# ==========================================

def evaluate_model(y_true, predictions):

    mae = mean_absolute_error(y_true, predictions)

    mse = mean_squared_error(y_true, predictions)

    rmse = mse ** 0.5

    r2 = r2_score(y_true, predictions)

    return mae, rmse, r2


# ==========================================
# EVALUATE LINEAR REGRESSION
# ==========================================

linear_mae, linear_rmse, linear_r2 = evaluate_model(
    y_test,
    linear_predictions
)

print("\n===== LINEAR REGRESSION RESULTS =====")

print("MAE :", linear_mae)

print("RMSE:", linear_rmse)

print("R2 Score:", linear_r2)


# ==========================================
# EVALUATE RANDOM FOREST
# ==========================================

rf_mae, rf_rmse, rf_r2 = evaluate_model(
    y_test,
    rf_predictions
)

print("\n===== RANDOM FOREST RESULTS =====")

print("MAE :", rf_mae)

print("RMSE:", rf_rmse)

print("R2 Score:", rf_r2)


# ==========================================
# MODEL COMPARISON
# ==========================================

print("\n===== MODEL COMPARISON =====")

print("\nLinear Regression R2 Score:", linear_r2)

print("Random Forest R2 Score:", rf_r2)


# ==========================================
# SAVE BEST MODEL
# ==========================================

pickle.dump(rf_model, open("model.pkl", "wb"))

print("\n===== BEST MODEL SAVED AS model.pkl =====")