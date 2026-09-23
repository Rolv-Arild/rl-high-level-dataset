import logging
import os
import time
import warnings
from pathlib import Path

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


def send_to_whosbotting(replay_path: str | Path | os.PathLike, max_retries: int = 5, timeout: float = 30.0):
    """Sends a replay file to whosbotting.com for ML-based bot detection."""
    headers = {"Content-Type": "application/replay"}
    if RATE_LIMIT_BYPASS_KEY:
        headers["Bypass-Password"] = RATE_LIMIT_BYPASS_KEY

    with open(replay_path, "rb") as f:
        data = f.read()

    retries = 0
    while True:
        try:
            response = requests.post(
                WHOSBOTTING_URL + WHOSBOTTING_ENDPOINT,
                data=data,
                headers=headers,
                timeout=timeout,
            )
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                retries += 1
                if retries > max_retries:
                    logging.warning(f"  whosbotting.com rate limit retries exceeded ({max_retries})")
                    return None
                retry_header = response.headers.get("Retry-After") or response.headers.get("retry_after", 10)
                try:
                    t = int(retry_header)
                except (ValueError, TypeError):
                    t = 10
                logging.info(f"  Rate limited by whosbotting.com, waiting {t}s (retry {retries}/{max_retries})...")
                time.sleep(t)
            else:
                logging.warning(f"  whosbotting.com error: {response.status_code} - {response.text[:200]}")
                return None
        except requests.RequestException as e:
            retries += 1
            if retries > max_retries:
                logging.warning(f"  whosbotting.com request failed after {max_retries} retries: {e}")
                return None
            s = 2 ** retries
            logging.info(f"  whosbotting.com connection error ({e}), retrying in {s}s...")
            time.sleep(s)


def has_cheater(replay_path: str | Path | os.PathLike) -> bool:
    whosbotting_verdict = send_to_whosbotting(replay_path)
    if not whosbotting_verdict or not isinstance(whosbotting_verdict, dict):
        return False
    for player_result in whosbotting_verdict.get("player_results", []):
        if player_result.get("confidence_percent", 0.0) >= 0.5:
            return True
    return False
