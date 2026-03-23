import pandas as pd
import requests
from pandas import json_normalize


# retrieves player point lines and player position of live prizepicks lines
def retrieve_lines(stat_type="Points"):
    session = requests.Session()
    response = session.get(
        "https://partner-api.prizepicks.com/projections?single_stat=True&league_id=7&per_page=100000'",
    )
    print(response.status_code)

    df1 = json_normalize(response.json()["included"])
    df1 = df1[df1["type"] == "new_player"]

    df2 = json_normalize(response.json()["data"])
    
    # Filter out Demon and Goblin multipliers to only get the true "Standard" lines
    if "attributes.odds_type" in df2.columns:
        df2 = df2[df2["attributes.odds_type"] == "standard"]

    data = pd.merge(
        df1,
        df2,
        how="left",
        left_on=["id"],
        right_on=["relationships.new_player.data.id"],
    )
    df = data[
        [
            "attributes.name",
            "attributes.line_score",
            "attributes.stat_type",
            "attributes.position",
            "attributes.description",
        ]
    ]
    prizepicksDf = df[df["attributes.stat_type"] == stat_type]
    column_mapping = {
        "attributes.name": "PLAYER",
        "attributes.line_score": "LINE",
        "attributes.position": "POSITION",
        "attributes.description": "MATCHUP",
    }

    position_mapping = {
        "G": "Guard",
        "F": "Forward",
        "C": "Center",
        "G-F": "Guard-Forward",
        "F-G": "Forward-Guard",
        "C-F": "Center-Forward",
        "F-C": "Forward-Center",
    }

    # Rename columns and positions
    prizepicksDf = prizepicksDf.copy()
    prizepicksDf.rename(columns=column_mapping, inplace=True)
    prizepicksDf["POSITION"] = prizepicksDf["POSITION"].map(position_mapping)
    return prizepicksDf
