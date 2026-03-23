import pandas as pd
from pandas import json_normalize
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog, BoxScoreUsageV2
from nba_api.stats.endpoints import commonplayerinfo, commonallplayers, leaguegamelog, playerindex
from nba_api.stats.endpoints import teamgamelog, scoreboardv2
import time
import os
import random
import re
import unicodedata

PROXY_LIST = os.getenv("PROXY_LIST", "").split(",")
PROXY_LIST = [p.strip() for p in PROXY_LIST if p.strip()]

def get_proxy():
    return random.choice(PROXY_LIST) if PROXY_LIST else None


_ACTIVE_PLAYERS_DF = None
_LEAGUE_GAME_LOG_DF = None
_PLAYER_INDEX_DF = None
_TODAY_SCORES_DICT = None

def _normalize_name(name):
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    name = re.sub(r'\s+(jr\.?|sr\.?|ii|iii|iv)$', '', name)
    name = re.sub(r"[.'-]", '', name)
    return name.strip()

def _get_active_players_df():
    global _ACTIVE_PLAYERS_DF
    if _ACTIVE_PLAYERS_DF is None:
        _ACTIVE_PLAYERS_DF = commonallplayers.CommonAllPlayers(is_only_current_season=1, proxy=get_proxy()).get_data_frames()[0]
        _ACTIVE_PLAYERS_DF['NORM_NAME'] = _ACTIVE_PLAYERS_DF['DISPLAY_FIRST_LAST'].apply(_normalize_name)
    return _ACTIVE_PLAYERS_DF

def _get_league_game_log_df():
    global _LEAGUE_GAME_LOG_DF
    if _LEAGUE_GAME_LOG_DF is None:
        _LEAGUE_GAME_LOG_DF = leaguegamelog.LeagueGameLog(player_or_team_abbreviation='P', proxy=get_proxy()).get_data_frames()[0]
        # Pre-sort perfectly descending by date so subset queries are natively chronological!
        _LEAGUE_GAME_LOG_DF['GAME_DATE'] = pd.to_datetime(_LEAGUE_GAME_LOG_DF['GAME_DATE'])
        _LEAGUE_GAME_LOG_DF = _LEAGUE_GAME_LOG_DF.sort_values(by='GAME_DATE', ascending=False)
    return _LEAGUE_GAME_LOG_DF

def _get_today_scores_dict():
    global _TODAY_SCORES_DICT
    if _TODAY_SCORES_DICT is None:
        _TODAY_SCORES_DICT = scoreboardv2.ScoreboardV2(proxy=get_proxy()).get_dict()
    return _TODAY_SCORES_DICT

# Gets nba player's NBA.com ID using full name
def get_player_id(name: str):
    df = _get_active_players_df()
    
    # Try exact string match first
    match = df[df['DISPLAY_FIRST_LAST'].str.lower() == name.lower()]
    if not match.empty:
        return match.iloc[0]['PERSON_ID']
        
    # Try normalized fuzzy match (ignores Jr, Sr, punctuation, accents)
    norm_name = _normalize_name(name)
    match = df[df['NORM_NAME'] == norm_name]
    if not match.empty:
        return match.iloc[0]['PERSON_ID']
        
    player_info = players.find_players_by_full_name(name)
    if not player_info:
        return None
    return player_info[0]["id"]


def get_player_gamelog(player_id):
    df = _get_league_game_log_df()
    player_log = df[df['PLAYER_ID'] == int(player_id)].copy()
    if player_log.empty:
        return pd.DataFrame(columns=["MATCHUP", "PTS", "REB", "AST", "MIN", "HOME"])

    df_filtered = pd.DataFrame({
        "MATCHUP": player_log["MATCHUP"],
        "PTS": player_log["PTS"],
        "REB": player_log["REB"],
        "AST": player_log["AST"],
        "MIN": player_log["MIN"],
    })
    
    # Extract HOME directly from "vs." inside MATCHUP
    df_filtered["HOME"] = df_filtered["MATCHUP"].str.contains("vs.").astype(int)
    # Standardize the matchup text to just the opposition team abbreviation
    df_filtered["MATCHUP"] = df_filtered["MATCHUP"].str[-3:]
    return df_filtered.reset_index(drop=True)


def get_player_position(player_id):
    global _PLAYER_INDEX_DF
    if _PLAYER_INDEX_DF is None:
        _PLAYER_INDEX_DF = playerindex.PlayerIndex(proxy=get_proxy()).get_data_frames()[0]

    match = _PLAYER_INDEX_DF[_PLAYER_INDEX_DF["PERSON_ID"] == int(player_id)]
    if not match.empty:
        return match.iloc[0]["POSITION"]
    return "Position not available"

# Returns data with gamelog
def get_full_data(player_id):
    for attempt in range(3):
        try:
            data = get_player_gamelog(player_id)
            if data is not None and not data.empty:
                data["POSITION"] = get_player_position(player_id)
            return data
        except Exception as e:
            print(f"NBA API Timeout for {player_id}. Retrying... ({attempt+1}/3)")
            time.sleep(2)


# Returns a player's average stat in the previous 5 games
def get_last5_avg_stat(player_id, stat="MIN"):
    df = _get_league_game_log_df()
    player_log = df[df['PLAYER_ID'] == int(player_id)]
    if player_log.empty:
        return 0
    # Since df is sorted descending by GAME_DATE, head(5) is exactly the last 5 games!
    last_5_games = player_log.head(5)
    return float(last_5_games[stat].mean())


# Returns True if the player's next game is at home
def get_home(player_id):
    for attempt in range(3):
        try:
            df = _get_active_players_df()
            player_row = df[df['PERSON_ID'] == int(player_id)]
            if player_row.empty:
                return False
            team_id = player_row.iloc[0]['TEAM_ID']
            
            todayscores = _get_today_scores_dict()
            for game in todayscores["resultSets"][0]["rowSet"]:
                if game[6] == team_id or game[7] == team_id:
                    return game[6] == team_id
            return False
            
        except Exception as e:
            if attempt == 2:
                return False
            global _TODAY_SCORES_DICT
            _TODAY_SCORES_DICT = None
            time.sleep(2)


# Return a list of all active NBA players' names
def get_player_list():
    df = _get_active_players_df()
    return df['DISPLAY_FIRST_LAST'].tolist()


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
