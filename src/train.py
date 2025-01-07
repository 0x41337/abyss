import numpy as np
import pandas as pd

from data import df
from features import RSI, EMA, MACD, LogReturn, Volatility, BollingerBands
from settings import load_config

from export import save_model, save_model_json

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_, hyperparameters = load_config()

# Generate the indicators
df["EMA"] = EMA()
df["MACD"] = MACD()
df["RSI"] = RSI()
df["LogReturn"] = LogReturn()
df["Volatility"] = Volatility()
df["Upper Band"], df["Lower Band"] = BollingerBands()

# Define independent (X) and dependent (y) variables
X = df[
    [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "EMA",
        "MACD",
        "RSI",
        "LogReturn",
        "Volatility",
        "Upper Band",
        "Lower Band",
    ]
]
y = df["Close"]

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model
model = XGBRegressor(
    n_estimators=hyperparameters["XGBRegressor"]["n_estimators"],
    early_stopping_rounds=hyperparameters["XGBRegressor"]["early_stopping_rounds"],
    learning_rate=hyperparameters["XGBRegressor"]["learning_rate"],
    max_depth=hyperparameters["XGBRegressor"]["max_depth"],
    min_child_weight=hyperparameters["XGBRegressor"]["min_child_weight"],
    subsample=hyperparameters["XGBRegressor"]["subsample"],
    colsample_bytree=hyperparameters["XGBRegressor"]["colsample_bytree"],
    gamma=hyperparameters["XGBRegressor"]["gamma"],
    reg_alpha=hyperparameters["XGBRegressor"]["reg_alpha"],
    reg_lambda=hyperparameters["XGBRegressor"]["reg_lambda"],
)

# Train the model
try:
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
except Exception as error:
    print(f"Error during model training: {error}")
    exit()

# Make predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# View evaluation metrics
metrics = {"Metric": ["MAE", "RMSE", "R2"], "Value": [mae, rmse, r2]}
metrics_df = pd.DataFrame(metrics)

print("### Model Evaluation Metrics ###")
print(metrics_df.to_string(index=False))

# Show the importance of features
print("\n### Feature Importance ###")
importance = model.feature_importances_
feature_importance_df = pd.DataFrame(
    {"Feature": X.columns, "Importance": importance}
).sort_values(by="Importance", ascending=False)

print(feature_importance_df.to_string(index=False))

# Save the model and JSON file
try:
    save_model(model)
    save_model_json(model)
    print("\nModel and JSON files saved successfully!")
except Exception as e:
    print(f"Error saving model: {e}")
