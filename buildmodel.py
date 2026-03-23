import pandas as pd
import playerdata as dt
from sklearn.ensemble import RandomForestRegressor
import time
from tqdm import tqdm
from datetime import datetime
import os
import pickle


def buildmodel(target_stat="PTS"):
    # Saved files and player index
    today_date = datetime.now().strftime("%Y-%m-%d")
    gamelogs_filename = f"gamelogs_as_of_{today_date}.csv"
    model_filename = f"xgboost_fitted_{target_stat}_{today_date}.pkl"

    if os.path.exists("models/" + model_filename):
        print("model already built")
        return

    # Features to build model on:
    features = ["MATCHUP", "HOME", "PLAYER", "POSITION", "LAST_5_MIN", f"LAST_5_{target_stat}"]
    players = dt.get_player_list()

    # Sleep to prevent https timeouts
    time.sleep(0.6)
    data = pd.DataFrame(columns=["PLAYER", "POSITION", "PTS", "REB", "AST", "MATCHUP", "HOME", "MIN", "LAST_5_MIN", "LAST_5_PTS", "LAST_5_REB", "LAST_5_AST"])

    # Check if the data file already exists
    if os.path.exists(gamelogs_filename):
        print(f"Loading existing gamelogs data from {gamelogs_filename}")
        predictdf = pd.read_csv(gamelogs_filename)
    else:
        # Check for intermediate checkpoint resume file
        checkpoint_filename = f"gamelogs_checkpoint_{today_date}.csv"
        processed_players = set()
        if os.path.exists(checkpoint_filename):
            print(f"Resuming download from {checkpoint_filename}...")
            predictdf = pd.read_csv(checkpoint_filename)
            processed_players = set(predictdf["PLAYER"].unique())
        else:
            predictdf = pd.DataFrame(columns=["PLAYER", "POSITION", "PTS", "REB", "AST", "MATCHUP", "HOME", "MIN", "LAST_5_MIN", "LAST_5_PTS", "LAST_5_REB", "LAST_5_AST"])
            
        # Acquire data
        for i in tqdm(range(len(players)), desc="Fetching BoxScores"):
            p = players[i]
            if p in processed_players:
                continue
                
            player_id = dt.get_player_id(p)
            datalog = dt.get_full_data(player_id)
            
            # Reinstating crucial failsafe so the script doesn't crash on timeouts
            if datalog is None or datalog.empty:
                continue
                
            datalog["PLAYER"] = p
            
            # Create rolling features (shift by 1 to prevent data leakage)
            datalog = datalog.sort_index(ascending=False).reset_index(drop=True)
            datalog["LAST_5_MIN"] = datalog["MIN"].rolling(5, min_periods=1).mean().shift(1).fillna(datalog["MIN"].mean())
            datalog["LAST_5_PTS"] = datalog["PTS"].rolling(5, min_periods=1).mean().shift(1).fillna(0)
            datalog["LAST_5_REB"] = datalog["REB"].rolling(5, min_periods=1).mean().shift(1).fillna(0)
            datalog["LAST_5_AST"] = datalog["AST"].rolling(5, min_periods=1).mean().shift(1).fillna(0)

            # Append player to master dataframe
            predictdf = pd.concat([predictdf, datalog], ignore_index=True)
            
            # Save checkpoint every 20 players to disk
            if i % 20 == 0:
                predictdf.to_csv(checkpoint_filename, index=False)
            
            time.sleep(0.6)
            
        # Successfully finished all players, save final and remove checkpoint
        predictdf.to_csv(gamelogs_filename, index=False)
        if os.path.exists(checkpoint_filename):
            os.remove(checkpoint_filename)

    # One hot encode categorical data to use with XGBoost
    data_one_hot_encoded = pd.get_dummies(
        predictdf, columns=["PLAYER", "POSITION", "MATCHUP"], prefix="Category"
    )
    data_one_hot_encoded = data_one_hot_encoded.reindex(
        sorted(data_one_hot_encoded.columns), axis=1
    )
    data_one_hot_encoded["LAST_5_MIN"] = data_one_hot_encoded["LAST_5_MIN"].astype("float64")
    data_one_hot_encoded[f"LAST_5_{target_stat}"] = data_one_hot_encoded[f"LAST_5_{target_stat}"].astype("float64")
    data_one_hot_encoded["HOME"] = data_one_hot_encoded["HOME"].astype("float64")

    # Drop the actual game stats to prevent leakage, except target stat
    data_one_hot_encoded = data_one_hot_encoded.drop(["MIN", "PTS", "REB", "AST"], axis=1, errors="ignore")

    # Separate into features and value we are predicting
    y = predictdf[target_stat]
    X = data_one_hot_encoded

    # Build and fit model
    model = RandomForestRegressor(
        n_estimators=350,
        max_depth=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    # Save the trained model
    with open("models/" + model_filename, "wb") as model_file:
        pickle.dump(model, model_file)
