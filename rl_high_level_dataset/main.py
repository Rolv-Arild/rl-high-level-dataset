import argparse
import glob
import json
import logging
import math
import os
import shelve
import shutil
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Optional, Iterable, Set

import requests
import ballchasing as bc
from ballchasing.util import get_players, get_pid, is_standard_replay, ensure_sorted, deduplicate, mix_replay_iterators, \
    get_gameplay_duration
from requests import HTTPError
from requests.adapters import HTTPAdapter
from tqdm import tqdm

from rl_high_level_dataset.anomaly import is_anomalous_replay
from rl_high_level_dataset.encounters import get_encounter_stats, EncounterStats
from rl_high_level_dataset.whosbotting import (
    has_cheater,
    load_whosbotting_cache,
    check_cached_cheater,
    load_cheater_accounts,
    replay_has_known_cheater,
)

REPLAYS_PER_SEASON = int(os.getenv("REPLAYS_PER_SEASON", "100_000_000"))


def filter_valid_replays(replays: Iterator[dict]) -> Iterator[dict]:
    for replay in replays:
        is_standard, reason = is_standard_replay(replay)
        if is_standard:
            yield replay
        else:
            logging.info(
                f"Skipping replay {replay.get('id')} due to: {reason}"
            )


def score_and_filter_replays(replays: Iterator[dict], player_scores: dict,
                             threshold: float = 0.25,
                             cheater_accounts: Optional[Set[str]] = None) -> Iterator[tuple[dict, float]]:
    for replay in replays:
        if cheater_accounts and replay_has_known_cheater(replay, cheater_accounts):
            logging.debug(f"Skipping replay {replay.get('id')} due to known cheater account.")
            continue
        players = get_players(replay)
        if not players:
            continue
        tot = 0.0
        for player in players:
            pid = get_pid(player)
            if not pid or pid not in player_scores:
                logging.debug(
                    f"Skipping replay {replay.get('id')} due to player {pid} not being qualified."
                )
                break
            ps = player_scores[pid]
            if ps.prob_pro <= 0 or ps.prob_ssl <= 0:
                break
            tot += math.log(ps.prob_pro) + math.log(ps.prob_ssl)
        else:
            # Probability that players are both pro and SSL
            score = math.exp(tot / len(players))  # Geometric mean
            if score > threshold:
                yield replay, score


def iterate_replays_cached(
        replays: Optional[Iterator[dict]],
        cache_path: str,
        overwrite: bool = False,
        append: bool = False,
) -> Iterator[tuple[dict, bool]]:
    # If the cache file exists, and we aren't overwriting or appending, just read and return
    if os.path.exists(cache_path) and not overwrite and not append:
        with open(cache_path, "r") as reader:
            for line in reader:
                line = line.strip()
                if line:
                    yield json.loads(line), True
        return

    # Create the cache directory and prepare the temporary working file
    working_path = cache_path.replace(".jsonl", "_working.jsonl")
    assert working_path != cache_path, "Temporary file must not be the same as the original file"
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    mode = "a+" if append and not overwrite else "w"
    seen = set()
    with open(working_path, mode) as writer:
        if mode == "a+":
            # Read the existing cache and yield it
            writer.seek(0)
            for line in writer:
                line = line.strip()
                if not line:
                    continue
                replay = json.loads(line)
                yield replay, True
                seen.add(replay["id"])

        # Now write the new incoming replays
        if replays is not None:
            for replay in replays:
                if replay["id"] not in seen:
                    writer.write(f"{json.dumps(replay)}\n")
                    seen.add(replay["id"])
                    yield replay, False

    # Atomically replace the old cache with the new working file
    if os.path.exists(cache_path):
        os.remove(cache_path)
    os.rename(working_path, cache_path)


def shared_pipeline(replays: Iterator[dict], cache_path: str, result_path: Optional[str] = None) -> Iterator[dict]:
    # First cache of raw results
    replays = iterate_replays_cached(replays, cache_path)
    replays = (replay for replay, from_cache in replays)
    # Make sure the replays are valid, sorted and unique
    replays = filter_valid_replays(replays)
    replays = ensure_sorted(replays, sort_dir=bc.SortDir.ASCENDING, sort_by=bc.ReplaySortBy.REPLAY_DATE)
    replays = deduplicate(replays, check_dates=True)
    # Second cache for results
    if result_path:
        replays = iterate_replays_cached(replays, result_path)
        replays = (replay for replay, from_cache in replays)
    yield from replays


def get_ranked_replays(bc_api: bc.Api, season: str, cache_dir: str):
    ranked_replays = bc_api.get_replays(
        playlist=bc.Playlist.RANKED,
        min_rank=bc.Rank.GRAND_CHAMPION_1,  # Pros should never be below this, season reset would put them in GC
        sort_by=bc.ReplaySortBy.REPLAY_DATE,
        sort_dir=bc.SortDir.ASCENDING,
        count=REPLAYS_PER_SEASON,
        season=season,
    )
    ranked_cache_path = os.path.join(cache_dir, f"season_{season}", "ranked", "gc_plus.jsonl")
    ranked_result_path = ranked_cache_path.replace(".jsonl", "_results.jsonl")
    ranked_replays = shared_pipeline(ranked_replays, ranked_cache_path, ranked_result_path)
    yield from ranked_replays


def get_private_replays(bc_api: bc.Api, season: str, player_id: str, cache_dir: str):
    private_replays = bc_api.get_replays(
        playlist=[bc.Playlist.PRIVATE, bc.Playlist.OFFLINE],
        sort_by=bc.ReplaySortBy.REPLAY_DATE,
        sort_dir=bc.SortDir.ASCENDING,
        season=season,
        count=REPLAYS_PER_SEASON,
        player_id=player_id,
        disable_prefetch=True,  # Prevent way too many requests at once since we do this for each player
    )
    clean_pid = player_id.replace(":", "_").replace("*", "x")
    private_cache_path = os.path.join(cache_dir, f"season_{season}", "private", f"{clean_pid}.jsonl")
    private_result_path = private_cache_path.replace(".jsonl", "_results.jsonl")
    private_replays = shared_pipeline(private_replays, private_cache_path, private_result_path)
    yield from private_replays


def get_qualified_player_scores(
        encounter_stats: dict[str, EncounterStats],
        player_stats: dict[str, EncounterStats],
        or_threshold: float = 0.5,
        and_threshold: float = 0.1,
        cheater_accounts: Optional[Set[str]] = None,
) -> dict[str, EncounterStats]:
    """Calculates qualified player scores given encounter and self stats."""
    player_scores = {}
    for pid in encounter_stats:
        if cheater_accounts and pid in cheater_accounts:
            continue
        encounters = encounter_stats[pid]
        own = player_stats.get(pid, EncounterStats())
        others = encounters - own
        prob_pro_and_ssl = others.prob_pro * others.prob_ssl
        prob_pro_or_ssl = others.prob_pro + others.prob_ssl - prob_pro_and_ssl
        if encounters.pro_count > 0 and own.ssl_count > 0:
            if prob_pro_or_ssl > or_threshold and prob_pro_and_ssl > and_threshold:
                player_scores[pid] = others
    return player_scores


def collect_replays(bc_api: bc.Api, cache_dir: str, cheater_accounts: Optional[Set[str]] = None) -> Iterator[tuple[dict, float]]:
    # Free-to-play seasons to collect: f5 through f23
    seasons = [f"f{i}" for i in range(5, 24)]
    # Note on post-f21 seasons (f22, f23, ...):
    # EAC was implemented after Season 21, breaking BakkesMod rank reporting and auto-uploading.
    # Consequently, ranked replays from f22 onwards lack rank tags and cannot form a valid encounter graph.
    # We explicitly reuse Season 21 player scores for all post-f21 seasons.
    f21_scores: Optional[dict[str, EncounterStats]] = None

    for season in seasons:
        logging.critical(f"Collecting replays for season {season}...")
        counts = Counter()

        season_num = int(season.lstrip("f"))
        if season_num > 21:
            # Explicitly reuse Season 21 player scores for post-f21 replays
            if f21_scores is None:
                f21_encounters_file = os.path.join(cache_dir, "season_f21", "ranked", "encounters.json")
                if os.path.exists(f21_encounters_file):
                    with open(f21_encounters_file, "r") as f:
                        data = json.load(f)
                    enc_stats = {k: EncounterStats(**v) for k, v in data["encounters"].items()}
                    p_stats = {k: EncounterStats(**v) for k, v in data["players"].items()}
                    f21_scores = get_qualified_player_scores(enc_stats, p_stats, cheater_accounts=cheater_accounts)
                else:
                    raise RuntimeError("Cannot score post-f21 season without completed season_f21 cache!")
            player_scores = f21_scores
            logging.critical(f"Using Season f21 scores ({len(player_scores)} qualified players) for post-f21 season {season}.")
        else:
            encounters_file = os.path.join(cache_dir, f"season_{season}", "ranked", "encounters.json")
            if os.path.exists(encounters_file):
                with open(encounters_file, "r") as f:
                    data = json.load(f)
                encounter_stats = {k: EncounterStats(**v) for k, v in data["encounters"].items()}
                player_stats = {k: EncounterStats(**v) for k, v in data["players"].items()}
            else:
                # First pass to find all players who have played with pros in ranked
                ranked_replays = get_ranked_replays(bc_api, season, cache_dir)
                encounter_stats, player_stats, _ = get_encounter_stats(ranked_replays)
                os.makedirs(os.path.dirname(encounters_file), exist_ok=True)
                with open(encounters_file, "w") as f:
                    json.dump({
                        "encounters": {k: asdict(v) for k, v in encounter_stats.items()},
                        "players": {k: asdict(v) for k, v in player_stats.items()},
                    }, f, indent=2)

            player_scores = get_qualified_player_scores(
                encounter_stats, player_stats, cheater_accounts=cheater_accounts
            )
            logging.critical(f"Qualified {len(player_scores)} players for season {season}.")

            if season == "f21":
                f21_scores = player_scores

        # Second pass to get ranked replays. Should be cached now.
        ranked_replays = get_ranked_replays(bc_api, season, cache_dir)
        ranked_replays = score_and_filter_replays(ranked_replays, player_scores, cheater_accounts=cheater_accounts)
        for replay, score in ranked_replays:
            logging.info(f"Accepted ranked replay {replay['id']} with score {score:.4f}")
            yield replay, score
            counts[replay.get("playlist_id")] += 1

        # Get valid private replays for all the qualified players
        player_iterators = {}
        for pid in player_scores:
            private_replays = get_private_replays(bc_api, season, pid, cache_dir)
            player_iterators[pid] = private_replays
        private_replays = mix_replay_iterators(
            *player_iterators.values(),
            sort_by=bc.ReplaySortBy.REPLAY_DATE,
            sort_dir=bc.SortDir.ASCENDING,
        )
        private_replays = deduplicate(private_replays)  # Deduplication across all players
        private_replays = score_and_filter_replays(private_replays, player_scores, cheater_accounts=cheater_accounts)
        for replay, score in private_replays:
            logging.info(f"Accepted private replay {replay['id']} with score {score:.4f}")
            yield replay, score
            counts[replay.get("playlist_id")] += 1
        logging.critical(f"Collected {sum(counts.values())} replays for season {season}. "
                         f"({dict(counts.most_common())})")


def collect_replay_scores(bc_api: bc.Api, cache_dir: str, cheater_accounts: Optional[Set[str]] = None):
    # Collect replay scores
    scores_path = os.path.join(cache_dir, "scores.json")
    if os.path.exists(scores_path):
        with open(scores_path, "r") as f:
            scores = json.load(f)
        scores = {mode: data for mode, data in scores.items() if mode in ("1v1", "2v2", "3v3")}
        if cheater_accounts:
            removed_count = 0
            for mode in list(scores.keys()):
                for rid, rinfo in list(scores[mode].items()):
                    replay_players = set(rinfo.get("players", []))
                    if replay_players & cheater_accounts:
                        del scores[mode][rid]
                        removed_count += 1
            if removed_count:
                logging.critical(f"Filtered out {removed_count} cheater replays from cached scores.")
    else:
        scores = {}
        replays = collect_replays(bc_api, cache_dir, cheater_accounts=cheater_accounts)
        for replay, score in replays:
            players = get_players(replay)
            gameplay_duration = get_gameplay_duration(replay)
            mode = f"{len(players) // 2}v{len(players) // 2}"
            if mode not in ("1v1", "2v2", "3v3"):
                continue
            rid = replay["id"]
            scores.setdefault(mode, {})[rid] = {
                "id": rid,
                "score": score,
                "gameplay_duration": gameplay_duration,
                "players": [get_pid(player) for player in players],
            }
        with open(scores_path, "w") as f:
            json.dump(scores, f, indent=2)
    return scores


def select_replays(scores: dict) -> Iterator[str]:
    # Pick out replays based on scores until we run out of a mode
    mode_scores = {}
    mode_durations = {}
    mode_indices = {}
    target_ratios = {}
    for mode in scores:
        mode_scores[mode] = sorted(scores[mode].values(), key=lambda x: x["score"], reverse=True)
        mode_durations[mode] = 0
        mode_indices[mode] = 0
        target_ratios[mode] = 1
        # target_ratios[mode] = int(mode[0]) + 1  # e.g., "2v2" -> 3
    while True:
        mode = min(mode_scores, key=lambda k: mode_durations[k] / target_ratios[k])
        try:
            idx = mode_indices[mode]
            replay_info = mode_scores[mode][idx]
            mode_indices[mode] += 1
        except IndexError:
            logging.info(f"No more replays available for mode {mode}. "
                         f"Mode durations: {mode_durations}")
            for m in mode_scores:
                idx = mode_indices[m]
                last_replays = [r["id"] for r in mode_scores[m][idx - 10:idx]]
                logging.debug(f"Mode {m} has {len(mode_scores[m]) - idx} replays left. "
                              f"Last 10 included replays: {last_replays}")
            break
        mode_durations[mode] += replay_info["gameplay_duration"]  # * len(replay_info["players"])
        yield replay_info["id"]


def get_deep_replays(
        bc_api: bc.Api,
        replays: Iterable[str],
        shelf_path: str,
        workers: int = 4
) -> Iterator[dict]:
    # Open the shelf initially to identify which replays are missing
    with shelve.open(shelf_path) as shelf:
        uncached_replays = [rid for rid in replays if rid not in shelf]

    # Create an iterator over the uncached replays using tqdm to display progress
    it = tqdm(
        uncached_replays,
        desc="Fetching uncached deep replays",
        total=len(uncached_replays),
        unit="replay"
    )

    def fetch_replay(replay_id: str) -> tuple[str, Optional[dict]]:
        try:
            return replay_id, bc_api.get_replay(replay_id)
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                return replay_id, None
            raise

    # Download missing replays concurrently and save them as they complete
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for replay_id, replay in executor.map(fetch_replay, it):
            if replay is not None:
                # Open, write, and immediately close the shelf for each item to avoid corrupting it on exit
                with shelve.open(shelf_path) as shelf:
                    shelf[replay_id] = replay
            it.set_postfix(dict(id=replay_id, rate_limits=bc_api.rate_limit_count))

    # Finally, read and yield all the replays sequentially from the shelf
    with shelve.open(shelf_path) as shelf:
        for rid in replays:
            if rid in shelf:
                yield shelf[rid]


def main():
    cur_path = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default=os.path.join(cur_path, "..", "out", "high_level"))
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--whosbotting-cache", type=str, default=None)
    parser.add_argument("--enable-whosbotting", action="store_true", default=False,
                        help="Enable ML-based bot detection via whosbotting.com (currently disabled by default)")
    parser.add_argument("--cheaters-file", type=str, default=None)
    parser.add_argument("--disable-anomaly-filter", action="store_true", default=False,
                        help="Disable statistical and behavioral anomaly filtering for non-serious / AFK matches.")
    parser.add_argument("--out-path", type=str)
    args = parser.parse_args()

    api_key = os.getenv("BC_API_KEY")
    assert api_key, "BC_API_KEY not set"
    bc_api = bc.Api(api_key, proactive_rate_limit=True, print_on_rate_limit=True)
    bc_api._session.mount("https://", HTTPAdapter(pool_connections=100, pool_maxsize=100))
    base_dir = Path(args.base_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else base_dir / "cache"
    out_path = Path(args.out_path)

    base_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path.mkdir(parents=True, exist_ok=True)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARN)
    file_handler = logging.FileHandler(str(base_dir / "collect_replays.log"), mode="w")
    file_handler.setLevel(logging.DEBUG)
    logging.basicConfig(
        handlers=[console_handler, file_handler],
        level=logging.NOTSET,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Load known cheater accounts list (temporary workaround for whosbotting)
    cheaters_file = Path(args.cheaters_file) if args.cheaters_file else Path(cur_path).parent / "cheaters.txt"
    cheater_accounts = load_cheater_accounts(cheaters_file)
    if cheater_accounts:
        logging.critical(f"Loaded {len(cheater_accounts)} known cheater accounts from {cheaters_file}")

    logging.critical("Starting replay collection...")
    scores = collect_replay_scores(bc_api, str(cache_dir), cheater_accounts=cheater_accounts)
    logging.critical("Scores calculated.")
    logging.info(f"Ballchasing rate limit stats after scoring: {bc_api.rate_limit_stats}")

    selected_replays = list(select_replays(scores))
    logging.critical(f"Selected {len(selected_replays)} replays.")
    shelf_path = str(cache_dir / "deep_replays.shelve")

    # Get detailed versions of the selected replays, and re-filter with new info
    deep_replays = get_deep_replays(bc_api, selected_replays, shelf_path, workers=3)
    deep_replays = filter_valid_replays(deep_replays)
    deep_replays = deduplicate(deep_replays, check_dates=False)  # Just check IDs now that MatchGUID is available
    replay_ids = set()
    with tqdm(deep_replays, desc="Collecting deep replays.", total=len(selected_replays), unit="replay") as it:
        for replay in it:
            replay_ids.add(replay["id"])
            it.set_postfix(dict(id=replay["id"], date=replay["date"], rate_limits=bc_api.rate_limit_count))

    logging.info(f"Ballchasing rate limit stats after deep replays: {bc_api.rate_limit_stats}")

    # Re-filter any deep replays containing known cheater accounts
    if cheater_accounts:
        with shelve.open(shelf_path) as shelf:
            flagged = {
                rid for rid in replay_ids
                if rid in shelf and replay_has_known_cheater(shelf[rid], cheater_accounts)
            }
            if flagged:
                logging.critical(f"Filtered out {len(flagged)} deep replays matching accounts in {cheaters_file}")
                replay_ids -= flagged

    # Filter out anomalous replays (AFK, freestyle, non-serious, mutator lobbies)
    if not args.disable_anomaly_filter:
        with shelve.open(shelf_path) as shelf:
            anomalous = set()
            for mode, mode_scores in scores.items():
                for rid in mode_scores:
                    if rid in replay_ids and rid in shelf:
                        is_anom, reasons = is_anomalous_replay(shelf[rid], mode=mode)
                        if is_anom:
                            anomalous.add(rid)
                            logging.debug(f"Filtered anomalous replay {rid}: {reasons}")
            if anomalous:
                logging.critical(f"Filtered out {len(anomalous)} anomalous replays across modes.")
                replay_ids -= anomalous

    # New scores with valid replays, then rebalance
    scores = {
        mode: {rid: rinfo for rid, rinfo in mode_scores.items() if rid in replay_ids}
        for mode, mode_scores in scores.items()
    }
    selected_replays = list(select_replays(scores))
    replay_ids = set(selected_replays)

    logging.critical(f"Collected {len(replay_ids)} deep replays.")

    # Load cached whosbotting verdicts if file exists
    whosbotting_cache_path = Path(args.whosbotting_cache) if args.whosbotting_cache else cache_dir / "whosbotting.jsonl"
    whosbotting_cache = load_whosbotting_cache(whosbotting_cache_path) if whosbotting_cache_path.exists() else {}
    if whosbotting_cache:
        logging.critical(f"Loaded {len(whosbotting_cache)} cached whosbotting verdicts from {whosbotting_cache_path}")

    # Download replays and filter out cheaters in a single pass
    cheater_replays = set()
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        for mode in scores:
            mode_path = out_path / mode
            mode_path.mkdir(parents=True, exist_ok=True)
            existing = {
                p.stem for p in mode_path.glob("*.replay")
            }
            to_delete = existing - replay_ids
            logging.info(f"Deleting {len(to_delete)} replays that are no longer needed for mode {mode}.")
            for rid in to_delete:
                (mode_path / f"{rid}.replay").unlink(missing_ok=True)

            to_include = [rinfo for rinfo in scores[mode].values() if rinfo["id"] in replay_ids]
            for replay_info in tqdm(to_include, desc=f"Downloading replays for mode {mode}"):
                rid = replay_info["id"]
                replay_path = mode_path / f"{rid}.replay"

                # Check if already cached as cheater
                cached_is_cheater = check_cached_cheater(rid, cache=whosbotting_cache)
                if cached_is_cheater is True:
                    cheater_replays.add(rid)
                    (mode_path / f"{rid}.replay").unlink(missing_ok=True)
                    continue

                # If already on disk: keep clean, only send to whosbotting if explicitly enabled
                if replay_path.exists():
                    if args.enable_whosbotting and cached_is_cheater is not False:
                        if has_cheater(replay_path, cache_path=whosbotting_cache_path, replay_id=rid, cache=whosbotting_cache):
                            cheater_replays.add(rid)
                            replay_path.unlink(missing_ok=True)
                    continue

                # Replay not yet downloaded: download to temp
                tmp_path = tmp_dir / f"{rid}.replay"
                bc_api.download_replay(rid, tmp_path)  # Stream chunks directly to Path
                if args.enable_whosbotting and cached_is_cheater is not False:
                    if has_cheater(tmp_path, cache_path=whosbotting_cache_path, replay_id=rid, cache=whosbotting_cache):
                        cheater_replays.add(rid)
                        tmp_path.unlink(missing_ok=True)
                        continue
                shutil.move(tmp_path, replay_path)

    # Remove cheaters from the final selected replay set and re-balance
    if cheater_replays:
        logging.critical(f"Filtered out {len(cheater_replays)} cheater replays.")
        replay_ids -= cheater_replays
        scores = {
            mode: {rid: rinfo for rid, rinfo in mode_scores.items() if rid in replay_ids}
            for mode, mode_scores in scores.items()
        }
        selected_replays = list(select_replays(scores))
        replay_ids = set(selected_replays)

    downloaded_replays = set(out_path.glob("*/*.replay"))
    # Clean up any downloaded replays that were dropped in re-balancing
    for rpath in list(downloaded_replays):
        if rpath.stem not in replay_ids:
            rpath.unlink(missing_ok=True)
            downloaded_replays.remove(rpath)

    assert len(downloaded_replays) == len(replay_ids), (
        f"Mismatch between downloaded replays ({len(downloaded_replays)}) and selected IDs ({len(replay_ids)})!"
    )
    logging.critical(f"Replay download complete. {len(replay_ids)} replays verified. Creating metadata files.")

    # Finally, add metadata files with the included replays
    with shelve.open(shelf_path) as shelf:
        for mode in scores:
            to_include = list(rinfo for rinfo in scores[mode].values() if rinfo["id"] in replay_ids)
            metadata_path = out_path / mode / "metadata.json"
            with (open(metadata_path, "w") as f,
                  tqdm(to_include, desc=f"Creating metadata file for mode {mode}") as it):
                for replay_info in it:
                    rid = replay_info["id"]
                    deep = shelf[rid]
                    f.write(json.dumps({
                        **replay_info,
                        "data": deep,
                    }) + "\n")

    logging.info(f"Final Ballchasing rate limit stats: {bc_api.rate_limit_stats}")
    logging.critical("Finished.")


if __name__ == '__main__':
    main()
