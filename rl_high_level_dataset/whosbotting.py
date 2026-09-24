import json
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Optional, Dict, Any

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


def verdict_has_cheater(verdict: Optional[Dict[str, Any]], threshold: float = 0.5) -> bool:
    """Evaluates whether a whosbotting response dictionary contains a detected cheater."""
    if not verdict or not isinstance(verdict, dict):
        return False
    for player_result in verdict.get("player_results", []):
        if player_result.get("confidence_percent", 0.0) >= threshold:
            return True
    return False


def load_whosbotting_cache(cache_path: str | Path | os.PathLike) -> dict[str, dict]:
    """Loads cached whosbotting results from a .jsonl or .json file into memory."""
    cache_path = Path(cache_path)
    cache: dict[str, dict] = {}
    if not cache_path.exists():
        return cache

    if cache_path.suffix == ".jsonl":
        with open(cache_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    rid = entry.get("id")
                    if rid:
                        cache[rid] = entry.get("verdict", entry)
                except json.JSONDecodeError as e:
                    logging.warning(f"Error parsing {cache_path}:{line_no}: {e}")
    else:
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, dict):
                            cache[k] = v.get("verdict", v)
        except Exception as e:
            logging.warning(f"Failed to read whosbotting cache from {cache_path}: {e}")
    return cache


def save_whosbotting_cache_entry(
    cache_path: str | Path | os.PathLike,
    replay_id: str,
    verdict: dict,
    cache: Optional[dict[str, dict]] = None,
    threshold: float = 0.5,
) -> None:
    """Appends/updates a cache entry for a replay in the cache file and memory dict."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache is not None:
        cache[replay_id] = verdict

    has_cheat = verdict_has_cheater(verdict, threshold=threshold)
    entry = {
        "id": replay_id,
        "has_cheater": has_cheat,
        "verdict": verdict,
    }

    if cache_path.suffix == ".jsonl":
        with open(cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    else:
        existing: dict = {}
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        existing[replay_id] = entry
        tmp = cache_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
        tmp.replace(cache_path)


def check_cached_cheater(
    replay_id: str,
    cache_path: Optional[str | Path | os.PathLike] = None,
    cache: Optional[dict[str, dict]] = None,
    threshold: float = 0.5,
) -> Optional[bool]:
    """
    Check if a replay ID has a known verdict in cache without requiring the replay file.
    Returns:
        True: Known cheater
        False: Known clean
        None: Not found in cache
    """
    if cache is not None:
        if replay_id in cache:
            return verdict_has_cheater(cache[replay_id], threshold=threshold)
        return None

    cache_file = cache_path or os.environ.get("WHOSBOTTING_CACHE_FILE")
    if cache_file and Path(cache_file).exists():
        file_cache = load_whosbotting_cache(cache_file)
        if replay_id in file_cache:
            return verdict_has_cheater(file_cache[replay_id], threshold=threshold)

    return None


def send_to_whosbotting(
    replay_path: str | Path | os.PathLike,
    max_retries: int = 5,
    timeout: float = 30.0,
    cache_path: Optional[str | Path | os.PathLike] = None,
    replay_id: Optional[str] = None,
    cache: Optional[dict[str, dict]] = None,
) -> Optional[dict]:
    """Sends a replay file to whosbotting.com for ML-based bot detection, checking and updating cache if configured."""
    cache_file = cache_path or os.environ.get("WHOSBOTTING_CACHE_FILE")
    rid = replay_id or Path(replay_path).stem

    # Check cache first
    if cache is not None and rid in cache:
        return cache[rid]

    if cache_file and Path(cache_file).exists() and cache is None:
        file_cache = load_whosbotting_cache(cache_file)
        if rid in file_cache:
            return file_cache[rid]

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
                verdict = response.json()
                if cache_file and rid:
                    save_whosbotting_cache_entry(cache_file, rid, verdict, cache=cache)
                elif cache is not None and rid:
                    cache[rid] = verdict
                return verdict
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


def has_cheater(
    replay_path: str | Path | os.PathLike,
    cache_path: Optional[str | Path | os.PathLike] = None,
    replay_id: Optional[str] = None,
    cache: Optional[dict[str, dict]] = None,
    threshold: float = 0.5,
) -> bool:
    """Checks whether a replay has a cheater, checking/updating cache."""
    rid = replay_id or Path(replay_path).stem
    if cache is not None and rid in cache:
        return verdict_has_cheater(cache[rid], threshold=threshold)

    verdict = send_to_whosbotting(
        replay_path,
        cache_path=cache_path,
        replay_id=rid,
        cache=cache,
    )
    return verdict_has_cheater(verdict, threshold=threshold)
