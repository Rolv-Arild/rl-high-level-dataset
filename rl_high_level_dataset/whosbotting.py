import logging
import os
import time
import warnings

import requests

RATE_LIMIT_BYPASS_KEY = os.environ.get("RATE_LIMIT_BYPASS_KEY")
if RATE_LIMIT_BYPASS_KEY is None:
    warnings.warn("RATE_LIMIT_BYPASS_KEY not set. You may hit rate limits when sending replays to whosbotting.com.")
WHOSBOTTING_URL = "https://whosbotting.com"
WHOSBOTTING_ENDPOINT = "/analyze"

RANKED_PLAYLISTS = ["ranked-duels", "ranked-doubles", "ranked-standard"]

WHOSBOTTING_PLATFORM_MAP = {
    # whosbotting -> ballchasing
    "steam": "steam",
    "epic games": "epic",
    "playstation": "ps4",
    "xbox": "xbox",
    "psynet": "psynet",
}


def send_to_whosbotting(replay_path):
    """Sends a replay file to whosbotting.com for ML-based bot detection."""
    headers = {"Content-Type": "application/replay"}
    if RATE_LIMIT_BYPASS_KEY:
        headers["Bypass-Password"] = RATE_LIMIT_BYPASS_KEY

    with open(replay_path, "rb") as f:
        data = f.read()

    response = requests.post(WHOSBOTTING_URL + WHOSBOTTING_ENDPOINT, data=data, headers=headers)
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 429:
        try:
            t = int(response.headers.get("retry_after", 10))
        except (ValueError, TypeError):
            t = 10
        logging.info(f"  Rate limited by whosbotting.com, waiting {t}s...")
        time.sleep(t)
        return send_to_whosbotting(replay_path)  # Retry
    else:
        logging.warning(f"  whosbotting.com error: {response.status_code} - {response.text[:200]}")
        return None


def has_cheater(replay_path):
    whosbotting_verdict = send_to_whosbotting(replay_path)
    if whosbotting_verdict is None:
        return False
    for player_result in whosbotting_verdict["player_results"]:
        if player_result["confidence_percent"] >= 0.5:
            return True
    return False
