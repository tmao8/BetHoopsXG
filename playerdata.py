import pandas as pd
import numpy as np
import requests
from pandas import json_normalize
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog


def get_player_gamelog(name: str):
    player_info = players.find_players_by_full_name(name)
    player_id = player_info[0]["id"]
    player_log = playergamelog.PlayerGameLog(player_id=player_id)
    player_log = player_log.get_data_frames()[0]
    df = pd.DataFrame(player_log[["MATCHUP", "PTS"]])
    df["MATCHUP"] = df["MATCHUP"].str[-3:]
    return df
