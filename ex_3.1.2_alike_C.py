from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import RidgeCV, LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
import numpy as np


data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled.std(axis=0))

baseline = LinearRegression().fit(X_train_scaled, y_train)
predictions = baseline.predict(X_test_scaled)

Ridge_model = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100]).fit(X_train_scaled, y_train)
Ridge_predictions = Ridge_model.predict(X_test_scaled)
Ridge_model_alpha = Ridge_model.alpha_
print(Ridge_model_alpha)

baseline_rmse = root_mean_squared_error(y_test, predictions)
Ridge_model_rmse = root_mean_squared_error(y_test, Ridge_predictions)

print(baseline_rmse, Ridge_model_rmse)

print("OLS coefs:  ", baseline.coef_)
print("Ridge coefs:", Ridge_model.coef_)
print("max abs diff:", np.max(np.abs(baseline.coef_ - Ridge_model.coef_)))

