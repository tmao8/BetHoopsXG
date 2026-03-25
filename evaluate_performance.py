import json
import os
import time
import unicodedata
from datetime import datetime, timedelta
from nba_api.stats.endpoints import leaguegamelog
from playerdata import get_proxy

HISTORY_FILE = "ui/public/api/history.json"
STATS = ["Points", "Rebounds", "Assists"]
STAT_TO_NBA = {"Points": "PTS", "Rebounds": "REB", "Assists": "AST"}


def normalize_name(name):
    """Normalize names: remove diacritics, handle Nic/Nicolas, suffixes, etc."""
    if not name:
        return ""
    # Remove diacritics
    name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    name = name.lower().strip()
    # Handle known variations
    name = name.replace("nicolas claxton", "nic claxton")
    # Remove common suffixes and punctuation
    for suffix in [" jr.", " sr.", " iii", " ii", " iv", "."]:
        name = name.replace(suffix, "")
    return name


def load_history():
    """Load existing history or initialize an empty list."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []


def save_history(history):
    """Persist history back to disk."""
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def fetch_yesterday_actuals():
    """
    Pull all player box scores from yesterday via LeagueGameLog.
    Returns a dict: { "PLAYER_NAME": {"PTS": x, "REB": y, "AST": z} }
    """
    yesterday_dt = datetime.now() - timedelta(days=1)
    yesterday = yesterday_dt.strftime("%m/%d/%Y")
    
    # Dynamically determine the season string (e.g., 2025-26)
    if yesterday_dt.month >= 10: # Season starts in October
        season = f"{yesterday_dt.year}-{str(yesterday_dt.year + 1)[2:]}"
    else:
        season = f"{yesterday_dt.year - 1}-{str(yesterday_dt.year)[2:]}"

    # NBA API fetch
    try:
        log = leaguegamelog.LeagueGameLog(
            season=season,
            date_from_nullable=yesterday,
            date_to_nullable=yesterday,
            player_or_team_abbreviation='P',
            proxy=get_proxy(),
            timeout=60
        )
        df = log.get_data_frames()[0]
        
        actuals = {}
        for _, row in df.iterrows():
            name = normalize_name(row["PLAYER_NAME"])
            actuals[name] = {
                "PTS": row["PTS"],
                "REB": row["REB"],
                "AST": row["AST"],
            }
        return actuals
    except Exception as e:
        print(f"❌ Failed to fetch yesterday's box scores: {e}")
        return {}


def grade_predictions(actuals):
    """
    For each stat category, load yesterday's prediction JSON,
    compare against actuals, and return a list of graded entries.
    """
    yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    graded = []

    for stat in STATS:
        prediction_file = f"ui/public/api/latest_{stat}.json"
        if not os.path.exists(prediction_file):
            continue

        with open(prediction_file, "r") as f:
            data = json.load(f)

        if data.get("status") != "success":
            continue

        nba_stat = STAT_TO_NBA[stat]

        for entry in data["data"]:
            player = entry["Player"]
            norm_player = normalize_name(player)
            line = float(entry["Line"])
            predicted = float(entry["Predicted"])
            matchup = entry.get("MATCHUP", "")

            # Determine our bet direction
            bet = "Over" if predicted > line else "Under"
            delta = abs(predicted - line)

            # Look up the actual stat
            actual = None
            if norm_player in actuals:
                actual = actuals[norm_player].get(nba_stat)

            # Cannot grade without an actual score
            if actual is None:
                continue

            # Determine win/loss
            if bet == "Over":
                won = actual > line
            else:
                won = actual < line

            graded.append({
                "date": yesterday_str,
                "player": player,
                "matchup": matchup,
                "stat": stat,
                "line": line,
                "predicted": round(predicted, 1),
                "actual": actual,
                "bet": bet,
                "delta": round(delta, 1),
                "won": won,
            })

    return graded


def run():
    """Main entry point."""
    history = load_history()

    # Check if we already graded yesterday
    yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    existing_dates = {entry["date"] for entry in history}
    if yesterday_str in existing_dates:
        print(f"Already graded {yesterday_str}. Skipping.")
        return

    print(f"Grading predictions for {yesterday_str}...")
    actuals = fetch_yesterday_actuals()

    if not actuals:
        print("No box scores found for yesterday. Skipping evaluation.")
        return

    graded = grade_predictions(actuals)
    if not graded:
        print("No predictions could be matched to actuals. Skipping.")
        return

    history.extend(graded)
    save_history(history)

    # Print summary
    wins = sum(1 for g in graded if g["won"])
    total = len(graded)
    high_conf = [g for g in graded if g["delta"] >= 1.0]
    hc_wins = sum(1 for g in high_conf if g["won"])

    print(f"✅ Graded {total} predictions for {yesterday_str}")
    print(f"   Overall: {wins}/{total} ({100*wins/total:.0f}%)")
    if high_conf:
        print(f"   High-Confidence (Δ≥1): {hc_wins}/{len(high_conf)} ({100*hc_wins/len(high_conf):.0f}%)")


if __name__ == "__main__":
    run()
