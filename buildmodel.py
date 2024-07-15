import pandas as pd
import playerdata as dt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from datetime import datetime
import os
import pickle
from sklearn.model_selection import GridSearchCV


def buildmodel():
    # Saved files and player index
    today_date = datetime.now().strftime("%Y-%m-%d")
    data_filename = f"gamelogs_as_of_{today_date}.csv"
    model_filename = f"xgboost_fitted_{today_date}.pkl"

    if os.path.exists("models/" + model_filename):
        print("model already built")
        return

    # Features to build model on:
    features = ["MATCHUP", "HOME", "PLAYER", "MIN", "POSITION"]
    players = dt.get_player_list()

    # Sleep to prevent https timeouts
    time.sleep(0.6)
    data = pd.DataFrame(columns=["PLAYER", "POSITION", "PTS", "MATCHUP", "HOME", "MIN"])

    # Check if the data file already exists
    if os.path.exists(data_filename):
        # Load existing data file
        data = pd.read_csv(data_filename)
    else:
        # Initialize an empty DataFrame if the file doesn't exist
        data = pd.DataFrame(
            columns=["PLAYER", "POSITION", "PTS", "MATCHUP", "HOME", "MIN"]
        )
        # Acquire data
        for i in tqdm(range(len(players)), desc="Fetching BoxScores"):
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
    data_one_hot_encoded = data_one_hot_encoded.reindex(
        sorted(data_one_hot_encoded.columns), axis=1
    )
    data_one_hot_encoded["MIN"] = data_one_hot_encoded["MIN"].astype("float64")
    data_one_hot_encoded["HOME"] = data_one_hot_encoded["HOME"].astype("float64")

    # Separate into features and value we are predicting
    y = data_one_hot_encoded["PTS"]
    X = data_one_hot_encoded.drop("PTS", axis=1)

    # Build and fit model
    model = xgb.XGBRegressor(
        enable_categorical=True,
        objective="reg:squarederror",
        tree_method="hist",
        learning_rate=0.17,
        subsample=0.5,
        reg_lambda=0.1,
        reg_alpha=0.1,
        n_estimators=350,
        colsample_bytree=0.5,
        gamma=0,
        max_depth=5,
        num_parallel_tree=10,
    )

    model.fit(X, y)

    # Save the trained model
    with open("models/" + model_filename, "wb") as model_file:
        pickle.dump(model, model_file)
