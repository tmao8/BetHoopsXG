import pandas as pd
import playerdata as dt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from datetime import datetime
import os
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from skopt import BayesSearchCV


# Saved files and player index
today_date = datetime.now().strftime("%Y-%m-%d")
data_filename = f"gamelogs_as_of_{today_date}.csv"

# Features to build model on:
features = ["MATCHUP", "HOME", "PLAYER", "MIN", "POSITION"]
players = dt.get_player_list()

# Sleep to prevent https timeouts
time.sleep(0.6)
data = pd.DataFrame(columns=["PLAYER", "POSITION", "PTS", "MATCHUP", "HOME", "MIN"])

# Use these with XGBoost enable_categorical (experimental)
# column_types = {
#     "PLAYER": "category",
#     "POSITION": "category",
#     "PTS": "int64",
#     "MATCHUP": "category",
#     "HOME": "bool",
#     "MIN": "float64",
# }

# Check if the data file already exists
if os.path.exists(data_filename):
    # Load existing data file
    data = pd.read_csv(data_filename, index_col=False)
else:
    # Initialize an empty DataFrame if the file doesn't exist
    data = pd.DataFrame(columns=["PLAYER", "POSITION", "PTS", "MATCHUP", "HOME", "MIN"])

    # Acquire data
    for i in tqdm(range(len(players))):
        p = players[i]
        player_id = dt.get_player_id(p)
        datalog = dt.get_full_data(player_id)
        datalog["PLAYER"] = p
        data = pd.concat([data, datalog])
        time.sleep(0.6)

    # Save data
    data.to_csv(data_filename, index=False)


# One hot encode categorical data to use with XGBoost
data_one_hot_encoded = pd.get_dummies(
    data, columns=["PLAYER", "POSITION", "MATCHUP"], prefix="Category"
)
# Sort columns to maintain consistency
data_one_hot_encoded = data_one_hot_encoded.reindex(
    sorted(data_one_hot_encoded.columns), axis=1
)
data_one_hot_encoded["MIN"] = data_one_hot_encoded["MIN"].astype("float64")
data_one_hot_encoded["HOME"] = data_one_hot_encoded["HOME"].astype("float64")


# Separate into features and value we are predicting
y = data_one_hot_encoded["PTS"]
X = data_one_hot_encoded.drop("PTS", axis=1)
X, y = shuffle(X, y, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# find best XGBoost params
param_grid = {
    "learning_rate": [0.01, 0.17],
    "n_estimators": [200, 250, 350],
    "max_depth": [1, 3, 5],
    "min_child_weight": [1, 2, 5, 6],
    "subsample": [0.4, 0.5, 0.6, 0.8, 1.0],
    "colsample_bytree": [0.4, 0.5, 0.6, 0.8, 1.0],
    "gamma": [0, 0.05, 0.1, 0.15, 0.2],
    "reg_alpha": [0, 0.05, 0.1, 0.15, 0.2],
    "reg_lambda": [0, 0.05, 0.1, 0.15, 0.2],
}

model = xgb.XGBRegressor(
    enable_categorical=True,
    objective="reg:squarederror",
    tree_method="hist",
    learning_rate=0.17,
    subsample=0.4,
    reg_lambda=0.05,
    reg_alpha=0.1,
    n_estimators=250,
    colsample_bytree=0.6,
    gamma=0.15,
    max_depth=5,
    min_child_weight=2,
    num_parallel_tree=10,
)

model.fit(X_train, y_train)

from sklearn.model_selection import RandomizedSearchCV

# # Create RandomizedSearchCV object
# random_search = RandomizedSearchCV(
#     model,
#     param_distributions=param_grid,
#     n_iter=500,
#     cv=2,
#     scoring="neg_mean_squared_error",
#     verbose=1,
# )

# # Fit the model
# random_search.fit(X_train, y_train)
# # Print best parameters
# print("\nBest Parameters:", random_search.best_params_)
# # Evaluate the best model on the test set
# y_pred = random_search.best_estimator_.predict(X_test)
# test_mse = mean_squared_error(y_test, y_pred)
# print(f"\nTest MSE for Best Model: {test_mse:.4f}")

# # Use GridSearchCV for hyperparameter tuning
# grid_search = GridSearchCV(
#     model, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1
# )
# grid_search.fit(X_train, y_train)
# print("Best Hyperparameters:", grid_search.best_params_)
# best_model = grid_search.best_estimator_
# best_predictions = best_model.predict(X_test)
# best_mse = mean_squared_error(y_test, best_predictions)
# print(f"Mean Squared Error: {best_mse}")

# Make predictions on the test set
predictions = model.predict(X_test)
comparison_df = pd.DataFrame(
    {
        # "PLAYER": X_test["PLAYER"],
        # "MATCHUP": X_test["MATCHUP"],
        # "POSITION": X_test["POSITION"],
        # "MIN": X_test["MIN"],
        # "HOME": X_test["HOME"],
        "Actual": y_test,
        "Predicted": predictions,
    }
)
comparison_df.to_csv("model_test.csv")

xgb.plot_importance(model, max_num_features=50)
plt.show()

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")
