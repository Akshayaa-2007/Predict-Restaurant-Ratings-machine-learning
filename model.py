import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score


# -----------------------------
# STEP 1: Load Dataset
# -----------------------------
data = pd.read_csv("Dataset .csv")

print("Dataset Loaded Successfully\n")
print(data.head())


# -----------------------------
# STEP 2: Handle Missing Values
# -----------------------------
for column in data.columns:
    if data[column].dtype == 'object':
        data[column].fillna(data[column].mode()[0], inplace=True)
    else:
        data[column].fillna(data[column].mean(), inplace=True)


# -----------------------------
# STEP 3: Encode Categorical Data
# -----------------------------
le = LabelEncoder()

for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = le.fit_transform(data[column])


# -----------------------------
# STEP 4: Define Features & Target
# -----------------------------
X = data.drop("Aggregate rating", axis=1)
y = data["Aggregate rating"]


# -----------------------------
# STEP 5: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------
# STEP 6: Train Models
# -----------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)


# -----------------------------
# STEP 7: Predictions
# -----------------------------
lr_pred = lr_model.predict(X_test)
dt_pred = dt_model.predict(X_test)


# -----------------------------
# STEP 8: Evaluation
# -----------------------------
print("\n----- Linear Regression -----")
print("MSE:", mean_squared_error(y_test, lr_pred))
print("R2 Score:", r2_score(y_test, lr_pred))

print("\n----- Decision Tree Regression -----")
print("MSE:", mean_squared_error(y_test, dt_pred))
print("R2 Score:", r2_score(y_test, dt_pred))


# -----------------------------
# STEP 9: Feature Importance
# -----------------------------
importance = pd.Series(lr_model.coef_, index=X.columns)
print("\nFeature Importance:")
print(importance.sort_values(ascending=False))
