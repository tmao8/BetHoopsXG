import pandas as pd
import requests
from pandas import json_normalize


# retrieves player point lines and player position of live prizepicks lines
def retrieve_point_lines():
    session = requests.Session()
    response = session.get(
        "https://partner-api.prizepicks.com/projections?single_stat=True&league_id=7&per_page=100000'",
    )
    print(response.status_code)

    df1 = json_normalize(response.json()["included"])
    df1 = df1[df1["type"] == "new_player"]

    df2 = json_normalize(response.json()["data"])

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
    prizepicksptsDf = df[df["attributes.stat_type"] == "Points"]
    column_mapping = {
        "attributes.name": "PLAYER",
        "attributes.line_score": "PTS",
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
    prizepicksptsDf = prizepicksptsDf.copy()
    prizepicksptsDf.rename(columns=column_mapping, inplace=True)
    prizepicksptsDf["POSITION"] = prizepicksptsDf["POSITION"].map(position_mapping)
    return prizepicksptsDf
