from datetime import datetime
import pandas as pd
import pickle
import prizepickslines as ppl
from tqdm import tqdm
import playerdata as dt
import buildmodel
import time
import os
import unicodedata


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
    long_stat = stat_mapping_reverse.get(target_stat, target_stat)
    
    short_stat_map = {
        "Points": "PTS",
        "Rebounds": "REB",
        "Assists": "AST"
    }
    short_stat = short_stat_map.get(long_stat, "PTS")
    
    predictdf = ppl.retrieve_lines(stat_type=long_stat)
    if "PLAYER" in predictdf.columns:
        predictdf["PLAYER"] = predictdf["PLAYER"].apply(lambda x: ''.join(c for c in unicodedata.normalize('NFD', x) if unicodedata.category(c) != 'Mn') if isinstance(x, str) else x)
    time.sleep(1)
    data_filename = f"gamelogs_as_of_{today_date}.csv"
    prizepicks_filename = f"prizepicks_{target_stat}_{today_date}.csv"

    # Check if the prizepicks data file already exists
    if os.path.exists(prizepicks_filename):
        # Load existing data file
        predictdf = pd.read_csv(prizepicks_filename, index_col=False)
        if "PLAYER" in predictdf.columns:
            predictdf["PLAYER"] = predictdf["PLAYER"].apply(lambda x: ''.join(c for c in unicodedata.normalize('NFD', x) if unicodedata.category(c) != 'Mn') if isinstance(x, str) else x)
        
        # If we added new features recently, delete and recreate
        if "LAST_5_MIN" not in predictdf.columns:
            os.remove(prizepicks_filename)
            predictdf = ppl.retrieve_lines(stat_type=long_stat)
            if "PLAYER" in predictdf.columns:
                predictdf["PLAYER"] = predictdf["PLAYER"].apply(lambda x: ''.join(c for c in unicodedata.normalize('NFD', x) if unicodedata.category(c) != 'Mn') if isinstance(x, str) else x)

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
            if row["PLAYER"] == "Tristan Silva":
                row["PLAYER"] = "Tristan da Silva"
                predictdf.at[index, "PLAYER"] = "Tristan da Silva"
            if row["PLAYER"] in playerset:
                predictdf = predictdf.drop(index)
            else:
                playerset.add(row["PLAYER"])
                player_id = dt.get_player_id(row["PLAYER"])
                
                if player_id is None:
                    print(f"⚠️ Could not find NBA ID for: {row['PLAYER']}. Please add them to the manual override list above.")
                    continue

                # Assign whether player is playing at home
                predictdf.at[index, "HOME"] = int(dt.get_home(player_id))

                # Assign estimated minutes for player
                predictdf.at[index, "LAST_5_MIN"] = dt.get_last5_avg_stat(player_id, "MIN")
                predictdf.at[index, f"LAST_5_{short_stat}"] = dt.get_last5_avg_stat(player_id, short_stat)
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

    predictdf = predictdf[X.columns].fillna(0)
    players = saveoriginal["PLAYER"]
    predictions = saved_model.predict(predictdf)
    comparison_df = pd.DataFrame(
        {
            "Player": players,
            "MATCHUP": saveoriginal["MATCHUP"],
            "Line": pointlines,
            "Predicted": predictions,
            "Stat": target_stat
        }
    )
    prediction_filename = f"model_predictions_{target_stat}_{today_date}.csv"
    comparison_df.to_csv(prediction_filename, index=False)
    return comparison_df
    # Save predictions
