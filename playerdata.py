import pandas as pd
from pandas import json_normalize
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog, BoxScoreUsageV2
from nba_api.stats.endpoints import commonplayerinfo
from nba_api.stats.endpoints import teamgamelog, scoreboardv2
import time
import os
import random

PROXY_LIST = os.getenv("PROXY_LIST", "").split(",")
PROXY_LIST = [p.strip() for p in PROXY_LIST if p.strip()]

def get_proxy():
    return random.choice(PROXY_LIST) if PROXY_LIST else None


# Gets nba player's NBA.com ID using full name
def get_player_id(name: str):
    player_info = players.find_players_by_full_name(name)
    return player_info[0]["id"]


def get_player_gamelog(player_id):
    # Get player id

    # Get player game log using id
    player_log = playergamelog.PlayerGameLog(player_id=player_id, proxy=get_proxy())
    player_log = player_log.get_data_frames()[0]

    # # Get Player Usage Rates:
    # game_ids = player_log["Game_ID"].tolist()
    # usage_rates = []
    # for game_id in game_ids:
    #     usage_data = BoxScoreUsageV2(game_id=game_id).get_data_frames()[0]
    #     usage_rate = usage_data.loc[
    #         usage_data["PLAYER_ID"] == player_id, "USG_PCT"
    #     ].values
    #     usage_rates.append(usage_rate[0])
    # player_log["USAGE_RATE"] = usage_rates

    # Filter data to include only team against and home game
    df = pd.DataFrame(player_log[["MATCHUP", "PTS", "REB", "AST", "MIN"]])
    df["HOME"] = (df["MATCHUP"].str[-5:-4] == ".").astype(int)
    df["MATCHUP"] = df["MATCHUP"].str[-3:]
    return df


def get_player_position(player_id):
    player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id, proxy=get_proxy())

    # Make the API request
    player_info_data = player_info.get_data_frames()[0]

    # Extract player position
    position = (
        player_info_data["POSITION"][0]
        if not player_info_data["POSITION"].empty
        else "Position not available"
    )

    return position


# Returns data with gamelog and position
def get_full_data(player_id):
    for attempt in range(3):
        try:
            pos = get_player_position(player_id)
            time.sleep(1.2) # Sleep between the position and gamelog requests to avoid rate limit timeout
            gamelog = get_player_gamelog(player_id)
            data = gamelog
            data["POSITION"] = pos
            return data
        except Exception as e:
            print(f"NBA API Timeout for {player_id}. Retrying... ({attempt+1}/3)")
            time.sleep(2)


# Returns a player's average stat in the previous 5 games
def get_last5_avg_stat(player_id, stat="MIN"):
    game_log = playergamelog.PlayerGameLog(player_id=player_id, proxy=get_proxy())
    game_log_data = game_log.get_data_frames()[0]
    # Select the last 5 games
    last_5_games = game_log_data.head(5)
    # Calculate the average
    avg_stat = last_5_games[stat].mean()
    return avg_stat


# Returns True if the player's next game is at home
def get_home(player_id):
    # Get the player's Team ID
    player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id, proxy=get_proxy())
    player_info_data = player_info.get_data_frames()[0]
    team_id = player_info_data["TEAM_ID"][0]
    todayscores = scoreboardv2.ScoreboardV2(proxy=get_proxy()).get_dict()
    for game in todayscores["resultSets"][0]["rowSet"]:
        if game[6] == team_id or game[7] == team_id:
            return game[6] == team_id
            break
    return False


# Returns full list of NBA players this season
def get_player_list():
    allactive = players.get_active_players()
    player_list = [nbaplayer["full_name"] for nbaplayer in allactive]
    return player_list


# Returns DF containing each player's point total in their most recent game
def get_last_game_pts(players):
    points = []
    for p in players:
        # Get player game log using id
        player_log = playergamelog.PlayerGameLog(player_id=get_player_id(p), proxy=get_proxy())
        player_log = player_log.get_data_frames()[0]
        last_game_points = player_log.iloc[0]["PTS"]
        points.append(last_game_points)
        time.sleep(0.6)
    df = pd.DataFrame(columns=["PTS"])
    df["PTS"] = points
    return df
