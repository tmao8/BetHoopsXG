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

# Saved files and player index
today_date = datetime.now().strftime("%Y-%m-%d")
data_filename = f"your_data_file_{today_date}.csv"
index_filename = f"last_index_{today_date}.txt"

# Features to build model on:
features = ["MATCHUP", "HOME", "PLAYER", "MIN", "POSITION"]
players = dt.get_player_list()

# Save data after every run: frequent https timeouts
time.sleep(0.6)
data = pd.DataFrame(columns=["PLAYER", "POSITION", "PTS", "MATCHUP", "HOME", "MIN"])

column_types = {
    "PLAYER": "category",
    "POSITION": "category",
    "PTS": "int64",
    "MATCHUP": "category",
    "HOME": "bool",
    "MIN": "int64",
}

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

data = data.astype(column_types)

X = data[features]
y = data["PTS"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = xgb.XGBRegressor(
    enable_categorical=True, objective="reg:squarederror", tree_method="hist"
)
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)
comparison_df = pd.DataFrame(
    {
        "PLAYER": X_test["PLAYER"],
        "MATCHUP": X_test["MATCHUP"],
        "Actual": y_test,
        "Predicted": predictions,
    }
)
comparison_df.to_csv("model_test.csv")

xgb.plot_importance(model, max_num_features=10)
plt.show()

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")
