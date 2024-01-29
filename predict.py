from datetime import datetime
import pandas as pd
import pickle
import prizepickslines as ppl
from tqdm import tqdm
import playerdata as dt
import time
import os

# Open saved (fitted) model
today_date = datetime.now().strftime("%Y-%m-%d")
model_filename = f"xgboost_fitted_{today_date}.pkl"
with open("models/" + model_filename, "rb") as model_file:
    saved_model = pickle.load(model_file)
predictdf = ppl.retrieve_point_lines()
time.sleep(5)
data_filename = f"gamelogs_as_of_{today_date}.csv"
prizepicks_filename = f"prizepicks_{today_date}.csv"

# Check if the prizepicks data file already exists
if os.path.exists(prizepicks_filename):
    # Load existing data file
    predictdf = pd.read_csv(prizepicks_filename, index_col=False)
else:
    playerset = set()
    # Fetch home/away + minutes data for players
    for index, row in tqdm(
        predictdf.iterrows(), total=len(predictdf), desc="Processing players"
    ):
        print(row["PLAYER"])
        # Fix error with certain player
        if row["PLAYER"] == "Nicolas Claxton":
            row["PLAYER"] = "Nic Claxton"
        if row["PLAYER"] in playerset:
            predictdf = predictdf.drop(index)
        else:
            playerset.add(row["PLAYER"])
            player_id = dt.get_player_id(row["PLAYER"])

            # Assign whether player is playing at home
            predictdf.at[index, "HOME"] = dt.get_home(player_id)
            time.sleep(0.6)

            # Assign estimated minutes for player
            predictdf.at[index, "MIN"] = dt.get_last5_avg_min(player_id)
            time.sleep(0.6)
            predictdf.to_csv(prizepicks_filename, index=False)

saveoriginal = predictdf.copy()
# Alter DF to one hot encode and have correct columns
pointlines = predictdf["PTS"]
predictdf = predictdf.drop("attributes.stat_type", axis=1)
predictdf = predictdf.drop("PTS", axis=1)
predictdf = pd.get_dummies(
    predictdf, columns=["PLAYER", "HOME", "POSITION", "MATCHUP"], prefix="Category"
)

# Get columns of full dataset
data = pd.read_csv(data_filename)
one_hot_encoded = pd.get_dummies(
    data, columns=["PLAYER", "HOME", "POSITION", "MATCHUP"], prefix="Category"
)
one_hot_encoded = one_hot_encoded.reindex(sorted(one_hot_encoded.columns), axis=1)
X = one_hot_encoded.drop("PTS", axis=1)


# Match columns of prediction set with full dataset
missing_columns = set(X.columns) - set(predictdf.columns)
predictdf = predictdf.reindex(
    columns=predictdf.columns.union(missing_columns), fill_value=0
)


players = saveoriginal["PLAYER"]
predictions = saved_model.predict(predictdf)
comparison_df = pd.DataFrame(
    {
        "Player": players,
        "Pointlines": pointlines,
        "Predicted": predictions,
    }
)

# Save predictions
prediction_filename = f"model_predictions{today_date}.csv"
comparison_df.to_csv(prediction_filename, index=False)
