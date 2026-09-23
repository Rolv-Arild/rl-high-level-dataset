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
from typing import Iterator, Optional

import ballchasing as bc
from ballchasing.util import get_players, get_pid, is_standard_replay, ensure_sorted, deduplicate, mix_replay_iterators, \
    get_gameplay_duration
from requests import HTTPError
from requests.adapters import HTTPAdapter
from tqdm import tqdm

from rl_high_level_dataset.encounters import get_encounter_stats
from rl_high_level_dataset.whosbotting import has_cheater

REPLAYS_PER_SEASON = int(os.getenv("REPLAYS_PER_SEASON", "100_000_000"))


def filter_valid_replays(replays: Iterator[dict]) -> Iterator[dict]:
    for replay in replays:
        is_standard, reason = is_standard_replay(replay)
        if is_standard:
            yield replay
        else:
            logging.info(
                f"Skipping replay {replay['id']} due to: {reason}"
            )


def score_and_filter_replays(replays: Iterator[dict], player_scores: dict,
                             threshold: float = 0.25) -> Iterator[tuple[dict, float]]:
    for replay in replays:
        players = get_players(replay)
        # tot = EncounterStats()
        tot = 0
        for player in players:
            pid = get_pid(player)
            if pid not in player_scores:
                logging.debug(
                    f"Skipping replay {replay['id']} due to player {pid} not being qualified."
                )
                break
            ps = player_scores[pid]
            # tot = tot + ps
            tot += math.log(ps.prob_pro) + math.log(ps.prob_ssl)
        else:
            # Probability that players are both pro and SSL
            # score = tot.prob_pro * tot.prob_ssl
            score = math.exp(tot / len(players))  # Geometric mean
            if score > threshold:
                yield replay, score


def iterate_replays_cached(
        replays: Iterator[dict],
        cache_path: str,
        overwrite: bool = False,
        append: bool = False,
) -> Iterator[tuple[dict, bool]]:
    # If the cache file exists, and we aren't overwriting or appending, just read and return
    if os.path.exists(cache_path) and not overwrite and not append:
        with open(cache_path, "r") as reader:
            for line in reader:
                yield json.loads(line), True
        return

    # Create the cache directory and prepare the temporary working file
    working_path = cache_path.replace(".jsonl", "_working.jsonl")
    assert working_path != cache_path, "Temporary file must not be the same as the original file"
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    # Open the working file in write mode ("w") so we can cleanly rebuild it
    mode = "a+" if append and not overwrite else "w"
    seen = set()
    with open(working_path, mode) as writer:
        if mode == "a+":
            # Read the existing cache and yield it
            writer.seek(0)
            for line in writer:
                replay = json.loads(line)
                yield replay, True
                seen.add(replay["id"])

        # Now write the new incoming replays
        for replay in replays:
            if replay["id"] not in seen:
                writer.write(f"{json.dumps(replay)}\n")
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
        season=season
    )
    ranked_cache_path = os.path.join(cache_dir, f"season_{season}", "ranked", f"gc_plus.jsonl")
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
        disable_prefetch=True,  # Prevent way too many request at once since we do this for each player
    )
    clean_pid = player_id.replace(":", "_").replace("*", "x")
    private_cache_path = os.path.join(cache_dir, f"season_{season}", "private", f"{clean_pid}.jsonl")
    private_result_path = private_cache_path.replace(".jsonl", "_results.jsonl")
    private_replays = shared_pipeline(private_replays, private_cache_path, private_result_path)
    yield from private_replays


def collect_replays(bc_api: bc.Api, cache_dir: str):
    seasons = bc.Season.FREE_TO_PLAY[4:21]  # f5-f21
    # Note that ballchasing.com has not updated seasons since f21.
    # Since EAC was implemented right after, breaking auto-uploads.
    # We intentionally do nothing to correct this, because scores from f22 onwards would be poor quality,
    # and we'd rather share scores between f21 and later seasons.
    # It won't include many ranked replays, but we can at least get private matches (i.e. RLCS)

    for season in seasons:
        logging.critical(f"Collecting replays for season {season}...")
        counts = Counter()

        # First, get ranked replays and calculate player scores
        ranked_replays = get_ranked_replays(bc_api, season, cache_dir)
        encounter_stats, player_stats, player_infos = get_encounter_stats(ranked_replays)

        # Every player must have at least these probabilities to ever be included
        or_threshold = 0.5  # of being pro OR ssl
        and_threshold = 0.1  # of being pro AND ssl

        with open(os.path.join(cache_dir, f"season_{season}", "ranked", "encounters.json"), "w") as f:
            json.dump({
                "encounters": {k: asdict(v) for k, v in encounter_stats.items()},
                "players": {k: asdict(v) for k, v in player_stats.items()},
            }, f, indent=2)
        player_scores = {}
        for pid in encounter_stats:
            encounters = encounter_stats[pid]
            own = player_stats[pid]
            others = encounters - own
            prob_pro_and_ssl = others.prob_pro * others.prob_ssl
            prob_pro_or_ssl = others.prob_pro + others.prob_ssl - prob_pro_and_ssl
            if encounters.pro_count > 0 and own.ssl_count > 0:
                if prob_pro_or_ssl > or_threshold:  # Qualification threshold
                    if prob_pro_and_ssl > and_threshold:
                        player_scores[pid] = others
        logging.critical(f"Qualified {len(player_scores)} players for season {season}.")

        # Second pass to get ranked replays. Should be cached now.
        ranked_replays = get_ranked_replays(bc_api, season, cache_dir)
        ranked_replays = score_and_filter_replays(ranked_replays, player_scores)
        for replay, score in ranked_replays:
            logging.info(f"Accepted ranked replay {replay['id']} with score {score:.4f}")
            yield replay, score
            counts[replay.get("playlist_id")] += 1

        # Get valid private replays for all the qualified players
        player_iterators = {}
        for qualified_player in player_scores.items():
            pid, score = qualified_player
            private_replays = get_private_replays(bc_api, season, pid, cache_dir)
            player_iterators[pid] = private_replays
        private_replays = mix_replay_iterators(
            *player_iterators.values(),
            sort_by=bc.ReplaySortBy.REPLAY_DATE,
            sort_dir=bc.SortDir.ASCENDING,
        )
        private_replays = deduplicate(private_replays)  # Deduplication across all players
        private_replays = score_and_filter_replays(private_replays, player_scores)
        for replay, score in private_replays:
            logging.info(f"Accepted private replay {replay['id']} with score {score:.4f}")
            yield replay, score
            counts[replay.get("playlist_id")] += 1
        logging.critical(f"Collected {sum(counts.values())} replays for season {season}. "
                         f"({dict(counts.most_common())}")


def collect_replay_scores(bc_api: bc.Api, cache_dir: str):
    # Collect replay scores
    scores_path = os.path.join(cache_dir, "scores.json")
    if os.path.exists(scores_path):
        with open(scores_path, "r") as f:
            scores = json.load(f)
    else:
        scores = {}
        replays = collect_replays(bc_api, cache_dir)
        for replay, score in replays:
            players = get_players(replay)
            gameplay_duration = get_gameplay_duration(replay)
            mode = f"{len(players) // 2}v{len(players) // 2}"
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


def get_deep_replays(bc_api: bc.Api, replays: Iterator[str | dict], shelf_path: str,
                     workers: int = 2, batch_size: int = 200) -> Iterator[dict]:
    with (ThreadPoolExecutor(max_workers=workers) as ex,
          shelve.open(shelf_path) as cache):
        futures = []
        for replay in replays:
            if isinstance(replay, str):
                rid = replay
            else:
                rid = replay["id"]
            if rid in cache:
                logging.debug(f"Using cached deep replay {rid}")
                yield cache[rid]
                continue
            f = ex.submit(bc_api.get_replay, rid)
            futures.append(f)
            while len(futures) >= batch_size:
                try:
                    res = futures.pop(0).result()
                    cache[res["id"]] = res
                    yield res
                except HTTPError:
                    continue
        while futures:
            try:
                res = futures.pop(0).result()
                cache[res["id"]] = res
                yield res
            except HTTPError:
                continue


def main():
    cur_path = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default=os.path.join(cur_path, "..", "out", "high_level"))
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--out-path", type=str)
    args = parser.parse_args()

    api_key = os.getenv("BC_API_KEY")
    assert api_key, "BC_API_KEY not set"
    bc_api = bc.Api(api_key)
    bc_api._session.mount("https://", HTTPAdapter(pool_connections=100, pool_maxsize=100))
    base_dir = args.base_dir
    cache_dir = args.cache_dir or os.path.join(base_dir, "cache")
    out_path = args.out_path

    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARN)
    file_handler = logging.FileHandler(os.path.join(base_dir, 'collect_replays.log'), mode='w')
    file_handler.setLevel(logging.DEBUG)
    logging.basicConfig(
        handlers=[console_handler, file_handler],
        level=logging.NOTSET,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logging.critical("Starting replay collection...")
    scores = collect_replay_scores(bc_api, cache_dir)
    logging.critical(f"Scores calculated.")
    selected_replays = list(select_replays(scores))
    logging.critical(f"Selected {len(selected_replays)} replays.")
    shelf_path = os.path.join(cache_dir, "deep_replays.shelve")

    # Get detailed versions of the selected replays, and re-filter with new info
    deep_replays = get_deep_replays(bc_api, selected_replays, shelf_path, workers=3)
    deep_replays = filter_valid_replays(deep_replays)
    deep_replays = deduplicate(deep_replays, check_dates=False)  # Just check IDs now that MatchGUID is available
    replay_ids = set()
    with tqdm(deep_replays, desc="Collecting deep replays.", total=len(selected_replays), unit="replay") as it:
        for replay in it:
            replay_ids.add(replay["id"])
            it.set_postfix(dict(id=replay["id"], date=replay["date"], rate_limits=bc_api.rate_limit_count))

    # New scores with valid replays, then rebalance
    scores = {
        mode: {rid: rinfo for rid, rinfo in mode_scores.items() if rid in replay_ids}
        for mode, mode_scores in scores.items()
    }
    selected_replays = list(select_replays(scores))
    replay_ids = set(selected_replays)

    logging.critical(f"Collected {len(replay_ids)} deep replays.")

    # Download replays and send to whosbotting.com for cheat detection
    cheater_replays = set()
    with tempfile.TemporaryDirectory() as tmp_dir:
        for mode in scores:
            mode_path = os.path.join(out_path, mode)
            os.makedirs(mode_path, exist_ok=True)
            existing = set(fn.replace(".replay", "")
                           for fn in os.listdir(mode_path)
                           if fn.endswith(".replay"))
            to_delete = existing - replay_ids
            logging.info(f"Deleting {len(to_delete)} replays that are no longer needed for mode {mode}.")
            for rid in to_delete:
                os.remove(os.path.join(mode_path, f"{rid}.replay"))

            to_include = list(rinfo for rinfo in scores[mode].values() if rinfo["id"] in replay_ids)
            for replay_info in tqdm(to_include, desc=f"Downloading replays for mode {mode}"):
                rid = replay_info["id"]
                folder = os.path.join(out_path, mode)
                os.makedirs(folder, exist_ok=True)
                replay_path = os.path.join(folder, f"{rid}.replay")
                if not os.path.exists(replay_path):
                    tmp_path = os.path.join(tmp_dir, f"{rid}.replay")
                    bc_api.download_replay(rid, tmp_path)  # Download to temp first
                    if has_cheater(tmp_path):
                        cheater_replays.add(rid)
                        continue
                    shutil.move(tmp_path, replay_path)

    # Remove cheater replays and do a final resample
    logging.critical("All replays checked for cheaters. Performing final resample.")
    replay_ids = replay_ids - cheater_replays
    scores = {
        mode: {rid: rinfo for rid, rinfo in mode_scores.items() if rid in replay_ids}
        for mode, mode_scores in scores.items()
    }
    selected_replays = list(select_replays(scores))
    replay_ids = set(selected_replays)
    downloaded_replays = set(glob.glob(os.path.join(out_path, "**", "*.replay"), recursive=True))
    found = 0
    # We should never get any new replays, only need to remove already downloaded ones
    for downloaded_replay in downloaded_replays:
        rid = os.path.splitext(os.path.basename(downloaded_replay))[0]
        if rid not in replay_ids:
            os.remove(downloaded_replay)
        else:
            found += 1
    assert found == len(replay_ids), "Some replays were not downloaded!"

    logging.critical(f"Final resample complete. {found} replays downloaded. Making metadata files.")

    # Finally, add metadata files with the included replays
    with shelve.open(shelf_path) as shelf:
        for mode in scores:
            to_include = list(rinfo for rinfo in scores[mode].values() if rinfo["id"] in replay_ids)
            metadata_path = os.path.join(out_path, mode, "metadata.json")
            with (open(metadata_path, "w") as f,
                  tqdm(to_include, desc=f"Creating metadata file for mode {mode}") as it):
                for replay_info in it:
                    rid = replay_info["id"]
                    deep = shelf[rid]
                    f.write(json.dumps({
                        **replay_info,
                        "data": deep,
                    }) + "\n")

    logging.critical("Finished.")


if __name__ == '__main__':
    main()
