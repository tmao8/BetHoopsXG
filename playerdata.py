import pandas as pd
import numpy as np
import requests
from pandas import json_normalize
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.endpoints import commonplayerinfo


# Gets nba player's NBA.com ID using full name
def get_player_id(name: str):
    player_info = players.find_players_by_full_name(name)
    return player_info[0]["id"]


def get_player_gamelog(player_id):
    # Get player id

    # Get player game log using id
    player_log = playergamelog.PlayerGameLog(player_id=player_id)
    player_log = player_log.get_data_frames()[0]

    # Filter data to include only team against and home game

    df = pd.DataFrame(player_log[["MATCHUP", "PTS", "MIN"]])
    df["HOME"] = df["MATCHUP"].str[-5:-4] == "."
    df["MATCHUP"] = df["MATCHUP"].str[-3:]
    return df


def get_player_position(player_id):
    player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)

    # Make the API request
    player_info_data = player_info.get_data_frames()[0]

    # Extract player position
    position = (
        player_info_data["POSITION"][0]
        if not player_info_data["POSITION"].empty
        else "Position not available"
    )

    return position


# returns data with gamelog and position
def get_full_data(player_id):
    pos = get_player_position(player_id)
    gamelog = get_player_gamelog(player_id)
    data = gamelog
    data["POSITION"] = pos
    return data


def get_player_list():
    allactive = players.get_active_players()
    player_list = [nbaplayer["full_name"] for nbaplayer in allactive]
    return player_list
