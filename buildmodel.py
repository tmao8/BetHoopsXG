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


# Saved files and player index
today_date = datetime.now().strftime("%Y-%m-%d")
data_filename = f"gamelogs_as_of_{today_date}.csv"
index_filename = f"last_index_{today_date}.txt"

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
    data = pd.DataFrame(columns=["PLAYER", "POSITION", "PTS", "MATCHUP", "HOME", "MIN"])

# Check if the index file already exists
if os.path.exists(index_filename):
    # Load existing index file
    with open(index_filename, "r") as file:
        last_index = int(file.read().strip())
else:
    # Set last_index to 0 if the file doesn't exist
    last_index = 0


# Acquire data in batches to help prevent timeouts
batch_size = 10

for i in tqdm(range(last_index, len(players), batch_size)):
    # Get a batch of players
    current_batch = players[i : i + batch_size]

    # Fetch data for the current batch
    for p in tqdm(current_batch):
        player_id = dt.get_player_id(p)
        datalog = dt.get_full_data(player_id)
        datalog["PLAYER"] = p
        data = pd.concat([data, datalog])
        time.sleep(0.6)

    # Save the last index in the index file
    with open(index_filename, "w") as file:
        file.write(str(i + batch_size))

    # Save data after each batch
    data.to_csv(data_filename, index=False)

# One hot encode categorical data to use with XGBoost
data_one_hot_encoded = pd.get_dummies(
    data, columns=["PLAYER", "HOME", "POSITION", "MATCHUP"], prefix="Category"
)
data_one_hot_encoded = data_one_hot_encoded.reindex(
    sorted(data_one_hot_encoded.columns), axis=1
)
data_one_hot_encoded["MIN"] = data_one_hot_encoded["MIN"].astype("float64")

# Separate into features and value we are predicting
y = data_one_hot_encoded["PTS"]
X = data_one_hot_encoded.drop("PTS", axis=1)


# Build and fit model
model = xgb.XGBRegressor(
    enable_categorical=True,
    objective="reg:squarederror",
    tree_method="hist",
    learning_rate=0.2,
    subsample=0.7,
    reg_lambda=0.1,
    reg_alpha=0.1,
    n_estimators=300,
    colsample_bytree=0.8,
    gamma=0,
    max_depth=4,
)

model.fit(X, y)

# Save the trained model
model_filename = f"xgboost_fitted_{today_date}.pkl"
with open("models/" + model_filename, "wb") as model_file:
    pickle.dump(model, model_file)
