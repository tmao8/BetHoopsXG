from datetime import datetime
import pandas as pd
import pickle
import prizepickslines as ppl
from tqdm import tqdm
import playerdata as dt
import buildmodel
import time
import os


def predict(target_stat="PTS"):
    # Open saved (fitted) model
    buildmodel.buildmodel(target_stat)
    today_date = datetime.now().strftime("%Y-%m-%d")
    model_filename = f"xgboost_fitted_{target_stat}_{today_date}.pkl"
    with open("models/" + model_filename, "rb") as model_file:
        saved_model = pickle.load(model_file)
        
    stat_mapping_reverse = {
        "PTS": "Points",
        "REB": "Rebounds",
        "AST": "Assists"
    }
    long_stat = stat_mapping_reverse.get(target_stat, "Points")
    predictdf = ppl.retrieve_lines(stat_type=long_stat)
    time.sleep(1)
    data_filename = f"gamelogs_as_of_{today_date}.csv"
    prizepicks_filename = f"prizepicks_{target_stat}_{today_date}.csv"

    # Check if the prizepicks data file already exists
    if os.path.exists(prizepicks_filename):
        # Load existing data file
        predictdf = pd.read_csv(prizepicks_filename, index_col=False)
        # If we added new features recently, delete and recreate
        if "LAST_5_MIN" not in predictdf.columns:
            os.remove(prizepicks_filename)
            predictdf = ppl.retrieve_lines(stat_type=long_stat)

    if not os.path.exists(prizepicks_filename):
        playerset = set()
        # Fetch home/away + minutes data for players
        for index, row in tqdm(
            predictdf.iterrows(), total=len(predictdf), desc="Processing players"
        ):
            # Fix error with certain player
            if row["PLAYER"] == "Nicolas Claxton":
                row["PLAYER"] = "Nic Claxton"
                predictdf.at[index, "PLAYER"] = "Nic Claxton"
            if row["PLAYER"] in playerset:
                predictdf = predictdf.drop(index)
            else:
                playerset.add(row["PLAYER"])
                player_id = dt.get_player_id(row["PLAYER"])

                # Assign whether player is playing at home
                predictdf.at[index, "HOME"] = dt.get_home(player_id).astype(int)
                time.sleep(0.6)

                # Assign estimated minutes for player
                predictdf.at[index, "LAST_5_MIN"] = dt.get_last5_avg_stat(player_id, "MIN")
                predictdf.at[index, f"LAST_5_{target_stat}"] = dt.get_last5_avg_stat(player_id, target_stat)
                time.sleep(0.6)
                predictdf.to_csv(prizepicks_filename, index=False)

    saveoriginal = predictdf.copy()
    # Alter DF to one hot encode and have correct columns
    pointlines = predictdf["LINE"]
    predictdf = predictdf.drop("attributes.stat_type", axis=1)
    predictdf = predictdf.drop("LINE", axis=1)
    predictdf = pd.get_dummies(
        predictdf, columns=["PLAYER", "POSITION", "MATCHUP"], prefix="Category"
    )

    # Get columns of full dataset
    data = pd.read_csv(data_filename)
    one_hot_encoded = pd.get_dummies(
        data, columns=["PLAYER", "POSITION", "MATCHUP"], prefix="Category"
    )
    one_hot_encoded = one_hot_encoded.reindex(sorted(one_hot_encoded.columns), axis=1)
    X = one_hot_encoded.drop(["MIN", "PTS", "REB", "AST"], axis=1, errors="ignore")

    # Match columns of prediction set with full dataset
    missing_columns = set(X.columns) - set(predictdf.columns)
    predictdf = predictdf.reindex(
        columns=list(predictdf.columns.union(missing_columns)), fill_value=0
    )

    predictdf = predictdf[X.columns]
    players = saveoriginal["PLAYER"]
    predictions = saved_model.predict(predictdf)
    comparison_df = pd.DataFrame(
        {
            "Player": players,
            "Line": pointlines,
            "Predicted": predictions,
            "Stat": target_stat
        }
    )
    prediction_filename = f"model_predictions_{target_stat}_{today_date}.csv"
    comparison_df.to_csv(prediction_filename, index=False)
    return comparison_df
    # Save predictions
