import os
import json
import csv
import re
import codecs

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Codes/ -> Src/
REPLAY_DATA_DIR = os.path.join(BASE_DIR, "Replay Data")

# Helper: robust JSON loader with encoding detection
def load_json_robust(path):
    # Try common encodings
    for enc in ["utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"]:
        try:
            with codecs.open(path, "r", encoding=enc) as f:
                text = f.read()
            # Remove anything before first '{'
            first_brace = text.find("{")
            if first_brace == -1:
                continue
            text = text[first_brace:]
            return json.loads(text)
        except Exception:
            continue
    raise ValueError("Could not decode JSON file with common encodings.")

# Loop through each Replay folder
for folder in os.listdir(REPLAY_DATA_DIR):
    folder_path = os.path.join(REPLAY_DATA_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    json_path = os.path.join(folder_path, "replay_summary.json")
    if not os.path.exists(json_path):
        print(f"[SKIP] No replay_summary.json in {folder}")
        continue

    # Load JSON robustly
    try:
        data = load_json_robust(json_path)
    except Exception as e:
        print(f"[ERROR] Could not read JSON in {folder}: {e}")
        continue

    # Extract goals
    goals = data.get("properties", {}).get("Goals", [])
    if not goals:
        print(f"[INFO] No goals in {folder}")
        continue

    # CSV path inside same folder
    csv_path = os.path.join(folder_path, "goals.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Team", "Frame", "PlayerName"])
        writer.writeheader()
        for g in goals:
            writer.writerow({
                "Team": g.get("PlayerTeam"),
                "Frame": g.get("frame"),
                "PlayerName": g.get("PlayerName")
            })

    print(f"[DONE] Extracted {len(goals)} goals → {csv_path}")
