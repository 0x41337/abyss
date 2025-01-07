from data import df
from features import RSI, EMA, MACD, LogReturn, Volatility, BollingerBands

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

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

# Defining the grid of hyperparameters for tuning
param_dist = {
    "n_estimators": [300, 400],
    "learning_rate": [0.001, 0.1, 0.2],
    "max_depth": [3, 4],
    "min_child_weight": [1, 3],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "gamma": [0, 0.1],
    "reg_alpha": [0.5, 1],
    "reg_lambda": [0.5, 1],
}

# Perform GridSearchCV with XGBRegressor
print("(This will take time!) Calculating hyperparameters combo to minimize NMSE...")
grid_search = GridSearchCV(
    XGBRegressor(), param_dist, cv=5, scoring="neg_mean_squared_error"
)

# Fit the model to the data
grid_search.fit(X_train, y_train)

# Output the best parameters and best score
print("Best set of hyperparameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)
