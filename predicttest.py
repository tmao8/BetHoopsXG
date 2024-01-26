from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm
import playerdata as dt

import os

yesterdays_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
prizepicks_filename = f"prizepicks_{yesterdays_date}.csv"
predictdf = pd.read_csv(prizepicks_filename, index_col=False)
player_list = predictdf["PLAYER"].tolist()
output_file = f"actual_pts_{yesterdays_date}.csv"
dt.get_last_game_pts(player_list).to_csv(output_file)
