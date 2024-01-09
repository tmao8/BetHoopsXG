import pandas as pd
import numpy as np
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
    return prizepicksptsDf


print(retrieve_point_lines())
