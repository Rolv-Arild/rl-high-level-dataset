"""
Anomaly detection for high-level Rocket League replays.

Detects non-serious matches, AFK players, self-imposed challenges (e.g. no powerslide),
freestyle lobbies, and mutator/throwing sessions using:
1. Mode-calibrated, per-minute player rate and positional stats (empirical percentiles).
2. Skellam distribution test for goal difference disparity over elapsed match time (scipy.stats.skellam).
3. Poisson process test for excessive goal frequencies (scipy.stats.poisson).
4. Exponential inter-arrival test for abnormally long overtimes (scipy.stats.expon).
5. Positional freestyle detection (turn-based ceiling/air setups from opposite nets).
6. Dedicated server awareness (RLCS dedicated server protection for legitimate pro matches).
"""

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable
from scipy import stats
from ballchasing.util import get_players, get_gameplay_duration


@dataclass(frozen=True)
class ModeRateThresholds:
    """Per-minute and positional thresholds for competitive play in a given mode."""
    min_bcpm: float              # Boost collected per minute floor
    min_bpm: float               # Boost consumed per minute floor
    min_avg_speed: float         # Average speed percentage floor (0-100)
    min_powerslide_pm: float     # Powerslides per minute floor (recovery / engagement)
    min_score_pm: float          # In-game score per minute floor (ball interaction)
    max_distance_to_ball: float  # Maximum average distance to ball in unreal units
    max_percent_infront: float   # Maximum percentage of time ahead of the ball (0-100)
    min_duration: float = 150.0  # Only apply maturity filters if gameplay duration >= min_duration


# Statistically derived from empirical distributions of >224,000 player samples
# in competitive ranked matches (tail percentiles <= 0.05% or >= 99.95%):
DEFAULT_MODE_THRESHOLDS: Dict[str, ModeRateThresholds] = {
    "1v1": ModeRateThresholds(
        min_bcpm=200.0,
        min_bpm=200.0,
        min_avg_speed=52.0,
        min_powerslide_pm=1.5,
        min_score_pm=30.0,
        max_distance_to_ball=3000.0,
        max_percent_infront=45.0,
        min_duration=150.0,
    ),
    "2v2": ModeRateThresholds(
        min_bcpm=180.0,
        min_bpm=180.0,
        min_avg_speed=50.0,
        min_powerslide_pm=1.5,
        min_score_pm=15.0,
        max_distance_to_ball=3600.0,
        max_percent_infront=45.0,
        min_duration=150.0,
    ),
    "3v3": ModeRateThresholds(
        min_bcpm=240.0,
        min_bpm=240.0,
        min_avg_speed=54.0,
        min_powerslide_pm=1.5,
        min_score_pm=12.0,
        max_distance_to_ball=3700.0,
        max_percent_infront=48.0,
        min_duration=150.0,
    ),
}

# Lenient floor thresholds for RLCS dedicated servers (only catch genuine disconnects / crashes / AFK)
RLCS_DISCONNECT_THRESHOLDS = ModeRateThresholds(
    min_bcpm=50.0,
    min_bpm=50.0,
    min_avg_speed=35.0,
    min_powerslide_pm=0.0,
    min_score_pm=2.0,
    max_distance_to_ball=0.0,
    max_percent_infront=0.0,
    min_duration=150.0,
)

# Empirical competitive baseline goal arrival intensity (goals per second):
DEFAULT_GOAL_RATES_PER_SEC: Dict[str, float] = {
    "1v1": 2.359 / 60.0,  # 1 goal every 25.4s
    "2v2": 1.241 / 60.0,  # 1 goal every 48.4s
    "3v3": 0.826 / 60.0,  # 1 goal every 72.6s
}


def is_rlcs_server(server_name: Optional[str]) -> bool:
    """Returns True if the server name indicates an RLCS dedicated tournament server."""
    return bool(server_name and server_name.lower().startswith("rlcs"))


# ==============================================================================
# Statistical Distribution Helpers (using scipy.stats)
# ==============================================================================

def skellam_diff_pvalue(diff: int, duration_sec: float, lambda_sec: float) -> float:
    """
    Two-sided exact Skellam test for goal difference under H0: equal Poisson arrival rates.
    mu_1 = mu_2 = (lambda_sec * duration_sec) / 2.
    """
    diff = abs(diff)
    if diff == 0:
        return 1.0
    mu = (lambda_sec * max(1.0, duration_sec)) / 2.0
    p_one_sided = stats.skellam.sf(diff - 1, mu, mu)
    return min(1.0, 2.0 * float(p_one_sided))


def poisson_sf(k: int, mu: float) -> float:
    """Calculates upper survival function P(X >= k) for Poisson(mu)."""
    if k <= 0:
        return 1.0
    if mu <= 0:
        return 0.0
    return float(stats.poisson.sf(k - 1, mu))


def exponential_overtime_pvalue(ot_seconds: float, lambda_per_sec: float) -> float:
    """
    Calculates P(T >= ot_seconds) for Exponential(lambda_per_sec).
    Represents the likelihood of sudden-death overtime lasting this long without a goal.
    """
    if ot_seconds <= 0 or lambda_per_sec <= 0:
        return 1.0
    return float(stats.expon.sf(ot_seconds, scale=1.0 / lambda_per_sec))


# ==============================================================================
# Player & Match Feature Extraction
# ==============================================================================

def extract_player_rate_stats(player: dict, duration_seconds: float) -> dict[str, float]:
    """
    Extracts per-minute and rate metrics for a player in a replay.
    All returned values are duration-normalized.
    """
    dur_minutes = max(0.5, duration_seconds) / 60.0
    stats_dict = player.get("stats", {})
    boost = stats_dict.get("boost", {})
    movement = stats_dict.get("movement", {})
    positioning = stats_dict.get("positioning", {})
    core = stats_dict.get("core", {})

    return {
        "bcpm": float(boost.get("bcpm", 0.0)),
        "bpm": float(boost.get("bpm", 0.0)),
        "avg_boost": float(boost.get("avg_amount", 0.0)),
        "avg_speed": float(movement.get("avg_speed_percentage", 0.0)),
        "percent_supersonic": float(movement.get("percent_supersonic_speed", 0.0)),
        "percent_slow": float(movement.get("percent_slow_speed", 0.0)),
        "powerslide_pm": float(movement.get("count_powerslide", 0)) / dur_minutes,
        "score_pm": float(core.get("score", 0)) / dur_minutes,
        "shots_pm": float(core.get("shots", 0)) / dur_minutes,
        "avg_distance_to_ball": float(positioning.get("avg_distance_to_ball", 0.0)),
        "percent_infront_ball": float(positioning.get("percent_infront_ball", 0.0)),
        "percent_high_air": float(movement.get("percent_high_air", 0.0)),
        "dist_no_possession": float(positioning.get("avg_distance_to_ball_no_possession", 0.0)),
    }


def check_player_anomalies(
    player: dict,
    duration_seconds: float,
    thresholds: ModeRateThresholds
) -> List[str]:
    """
    Checks if a player's duration-normalized stats violate the mode thresholds.
    Returns a list of violation descriptions.
    """
    pname = player.get("name", "Unknown")
    rates = extract_player_rate_stats(player, duration_seconds)
    violations = []

    # Hard physical activity floors (apply to matches of any length)
    if thresholds.min_bcpm > 0 and rates["bcpm"] < thresholds.min_bcpm:
        violations.append(f"Low BCPM ({pname}: {rates['bcpm']:.1f} < {thresholds.min_bcpm})")

    if thresholds.min_bpm > 0 and rates["bpm"] < thresholds.min_bpm:
        violations.append(f"Low BPM ({pname}: {rates['bpm']:.1f} < {thresholds.min_bpm})")

    if thresholds.min_avg_speed > 0 and rates["avg_speed"] < thresholds.min_avg_speed:
        violations.append(f"Low Avg Speed ({pname}: {rates['avg_speed']:.1f}% < {thresholds.min_avg_speed}%)")

    # Maturity rate and positioning filters (only applied if match has lasted long enough to settle)
    if duration_seconds >= thresholds.min_duration:
        if thresholds.min_powerslide_pm > 0 and rates["powerslide_pm"] < thresholds.min_powerslide_pm:
            violations.append(
                f"Low Powerslides ({pname}: {rates['powerslide_pm']:.1f}/min < {thresholds.min_powerslide_pm}/min)"
            )

        if thresholds.min_score_pm > 0 and rates["score_pm"] < thresholds.min_score_pm:
            violations.append(
                f"Low Score ({pname}: {rates['score_pm']:.1f}/min < {thresholds.min_score_pm}/min)"
            )

        if thresholds.max_distance_to_ball > 0 and rates["avg_distance_to_ball"] > thresholds.max_distance_to_ball:
            violations.append(
                f"Extreme Ball Distance ({pname}: {rates['avg_distance_to_ball']:.0f} > {thresholds.max_distance_to_ball:.0f} uu)"
            )

        if thresholds.max_percent_infront > 0 and rates["percent_infront_ball"] > thresholds.max_percent_infront:
            violations.append(
                f"High Time In Front of Ball ({pname}: {rates['percent_infront_ball']:.1f}% > {thresholds.max_percent_infront}%)"
            )

    return violations


def check_match_statistical_anomalies(
    deep: dict,
    mode: str,
    duration_seconds: float,
    p_skellam_threshold: float = 1e-4,
    p_poisson_threshold: float = 1e-4,
    p_overtime_threshold: float = 1e-5,
    enable_freestyle_check: bool = True,
) -> List[str]:
    """
    Applies match-level probability tests and freestyle detection across the entire match.
    """
    violations = []
    blue_players = deep.get("blue", {}).get("players", [])
    orange_players = deep.get("orange", {}).get("players", [])

    g_blue = sum(p.get("stats", {}).get("core", {}).get("goals", 0) for p in blue_players)
    g_orange = sum(p.get("stats", {}).get("core", {}).get("goals", 0) for p in orange_players)
    tot_goals = g_blue + g_orange
    goal_diff = abs(g_blue - g_orange)

    dur_sec = duration_seconds
    lam_sec = DEFAULT_GOAL_RATES_PER_SEC.get(mode, 2.359 / 60.0)

    # 1. Skellam test for skill disparity / intentional throwing (takes match duration into account)
    if goal_diff >= 4:
        p_skellam = skellam_diff_pvalue(goal_diff, dur_sec, lam_sec)
        if p_skellam < p_skellam_threshold:
            violations.append(
                f"Skellam goal disparity ({g_blue}-{g_orange} in {dur_sec:.0f}s, p={p_skellam:.2e} < {p_skellam_threshold:.0e})"
            )

    # 2. Poisson test for excessive goal frequency (kickoff / mutator spam)
    exp_goals = lam_sec * dur_sec
    p_poisson = poisson_sf(tot_goals, exp_goals)
    if p_poisson < p_poisson_threshold:
        violations.append(
            f"Excessive goal frequency ({tot_goals} goals in {dur_sec:.0f}s, exp={exp_goals:.1f}, p={p_poisson:.2e})"
        )

    # 3. Exponential test for sudden death overtime improbability (custom lobby match length)
    if dur_sec > 300 and (g_blue == g_orange + 1 or g_orange == g_blue + 1):
        ot_sec = dur_sec - 300
        p_ot = exponential_overtime_pvalue(ot_sec, lam_sec)
        if p_ot < p_overtime_threshold:
            tot_saves = sum(p.get("stats", {}).get("core", {}).get("saves", 0) for p in blue_players + orange_players)
            # Only flag if defensive saves are not exceptionally high (real competitive stalemates produce 20+ saves)
            if tot_saves < (ot_sec / 30.0):
                violations.append(
                    f"Improbable overtime duration ({ot_sec:.0f}s, p={p_ot:.2e} < {p_overtime_threshold:.0e})"
                )

    # 4. Freestyle 1v1 signature: turn-based airborne setups from opposite nets
    if enable_freestyle_check and mode == "1v1" and len(blue_players) == 1 and len(
            orange_players) == 1 and dur_sec >= 180:
        p1 = blue_players[0]
        p2 = orange_players[0]
        r1 = extract_player_rate_stats(p1, dur_sec)
        r2 = extract_player_rate_stats(p2, dur_sec)

        # Both players spend high time airborne, high time crawling slowly in net,
        # and give huge separation when not in possession.
        if (
                r1["percent_high_air"] >= 11.0 and r2["percent_high_air"] >= 11.0
                and r1["percent_slow"] >= 50.0 and r2["percent_slow"] >= 50.0
                and r1["dist_no_possession"] >= 3500.0 and r2["dist_no_possession"] >= 3500.0
        ):
            avg_air = (r1["percent_high_air"] + r2["percent_high_air"]) / 2.0
            avg_dist = (r1["dist_no_possession"] + r2["dist_no_possession"]) / 2.0
            violations.append(
                f"Freestyle 1v1 match (avg high air={avg_air:.1f}%, avg no-possession dist={avg_dist:.0f} uu)"
            )

    return violations


def is_anomalous_replay(
        replay: dict,
        mode: Optional[str] = None,
        custom_thresholds: Optional[Dict[str, ModeRateThresholds]] = None,
        enable_statistical_checks: bool = True,
        enable_freestyle_check: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Determines whether a replay is anomalous (AFK, non-competitive, freestyle, or trolling).

    Accepts either a deep replay dictionary or an entry dictionary containing 'data'.
    Returns (is_anomalous, reasons).
    """
    deep = replay.get("data", replay)
    players = get_players(deep)
    if not players:
        return False, []

    # Infer mode if not explicitly provided
    if mode is None:
        team_size = len(players) // 2
        mode = f"{team_size}v{team_size}"

    # Accurate gameplay clock duration
    duration_sec = float(replay.get("gameplay_duration") or get_gameplay_duration(deep) or 300.0)

    all_violations = []

    # Check if replay was hosted on an RLCS dedicated server
    server_name = (deep.get("server") or {}).get("name", "")
    if is_rlcs_server(server_name):
        # RLCS dedicated server match: protect legitimate pro games while checking for severe disconnects or corruption
        for player in players:
            if not player.get("stats"):
                continue
            p_violations = check_player_anomalies(player, duration_sec, RLCS_DISCONNECT_THRESHOLDS)
            all_violations.extend(p_violations)

        # Extreme corruption / goal spam check (e.g. 15 goals in 25s test matches)
        blue_players = deep.get("blue", {}).get("players", [])
        orange_players = deep.get("orange", {}).get("players", [])
        tot_goals = sum(p.get("stats", {}).get("core", {}).get("goals", 0) for p in blue_players + orange_players)
        lam_sec = DEFAULT_GOAL_RATES_PER_SEC.get(mode, 0.826 / 60.0)
        exp_goals = lam_sec * duration_sec
        p_poisson = poisson_sf(tot_goals, exp_goals)
        if p_poisson < 1e-6:
            all_violations.append(
                f"Corrupted match / extreme goal spam ({tot_goals} goals in {duration_sec:.0f}s, exp={exp_goals:.1f}, p={p_poisson:.2e})"
            )
        return (len(all_violations) > 0, all_violations)

    thresholds_map = custom_thresholds or DEFAULT_MODE_THRESHOLDS
    mode_th = thresholds_map.get(mode)
    if mode_th is None:
        return False, []

    # 1. Per-player rate & movement anomalies
    for player in players:
        if not player.get("stats"):
            continue
        p_violations = check_player_anomalies(player, duration_sec, mode_th)
        all_violations.extend(p_violations)

    # 2. Match-level probability & freestyle tests
    if enable_statistical_checks:
        m_violations = check_match_statistical_anomalies(
            deep, mode, duration_sec, enable_freestyle_check=enable_freestyle_check
        )
        all_violations.extend(m_violations)

    return (len(all_violations) > 0, all_violations)


def calculate_mode_percentiles(
        replays: Iterable[dict],
        lower_percentile: float = 0.05,
        upper_percentile: float = 99.95,
        min_duration: float = 150.0,
) -> Dict[str, Dict[str, float]]:
    """
    Calculates empirical distribution percentiles for per-minute metrics from an iterable of replays.
    """
    metrics: Dict[str, List[float]] = {
        "bcpm": [],
        "bpm": [],
        "avg_speed": [],
        "powerslide_pm": [],
        "score_pm": [],
        "avg_distance_to_ball": [],
        "percent_infront_ball": [],
    }

    for replay in replays:
        deep = replay.get("data", replay)
        dur = float(replay.get("gameplay_duration") or get_gameplay_duration(deep) or 300.0)
        if dur < min_duration:
            continue
        for player in get_players(deep):
            if not player.get("stats"):
                continue
            rates = extract_player_rate_stats(player, dur)
            for k in metrics:
                if rates.get(k) is not None:
                    metrics[k].append(rates[k])

    results: Dict[str, Dict[str, float]] = {}
    for k, vals in metrics.items():
        if not vals:
            continue
        vals.sort()
        n = len(vals)
        low_idx = max(0, min(n - 1, int(n * (lower_percentile / 100.0))))
        high_idx = max(0, min(n - 1, int(n * (upper_percentile / 100.0))))
        med_idx = max(0, min(n - 1, int(n * 0.5)))
        results[k] = {
            "lower": vals[low_idx],
            "median": vals[med_idx],
            "upper": vals[high_idx],
        }

    return results


def scan_and_report_dataset(
        dataset_dir: str,
        modes: Optional[List[str]] = None,
        prune: bool = False,
        output_report: Optional[str] = None
) -> dict:
    """
    Scans metadata files in a dataset directory, detects anomalous replays,
    and optionally deletes anomalous .replay files and cleans metadata.json.
    """
    target_modes = modes or ["1v1", "2v2", "3v3"]
    dataset_path = Path(dataset_dir)
    results = {}

    for mode in target_modes:
        mode_dir = dataset_path / mode
        meta_file = mode_dir / "metadata.json"
        if not meta_file.exists():
            continue

        anomalous_records = []
        valid_lines = []
        total_count = 0

        with open(meta_file, "r") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                total_count += 1
                record = json.loads(line_str)
                is_anom, reasons = is_anomalous_replay(record, mode=mode)
                if is_anom:
                    anomalous_records.append({
                        "id": record.get("id"),
                        "title": record.get("data", {}).get("title"),
                        "duration": record.get("gameplay_duration"),
                        "reasons": reasons,
                    })
                    if prune:
                        replay_file = mode_dir / f"{record.get('id')}.replay"
                        if replay_file.exists():
                            try:
                                replay_file.unlink()
                            except OSError as err:
                                logging.warning(f"Failed to delete {replay_file}: {err}")
                else:
                    valid_lines.append(line_str)

        if prune and anomalous_records:
            tmp_meta = mode_dir / "metadata.json.tmp"
            with open(tmp_meta, "w") as out_f:
                for vline in valid_lines:
                    out_f.write(vline + "\n")
            tmp_meta.replace(meta_file)

        results[mode] = {
            "total": total_count,
            "anomalous_count": len(anomalous_records),
            "anomalous_pct": (len(anomalous_records) / total_count * 100.0) if total_count else 0.0,
            "anomalies": anomalous_records,
        }

    if output_report:
        with open(output_report, "w") as rf:
            json.dump(results, rf, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Audit and filter anomalous Rocket League replays.")
    parser.add_argument("--dataset-dir", type=str, default="/mnt/disk1/rokutleg/replays/high_level",
                        help="Path to directory containing mode folders (1v1, 2v2, 3v3).")
    parser.add_argument("--modes", nargs="+", default=["1v1", "2v2", "3v3"],
                        help="Modes to evaluate.")
    parser.add_argument("--prune", action="store_true", default=False,
                        help="Delete anomalous .replay files and rewrite metadata.json.")
    parser.add_argument("--report", type=str, default=None,
                        help="Optional path to write a JSON summary report.")
    args = parser.parse_args()

    print(f"Scanning dataset in: {args.dataset_dir} (prune={args.prune})...")
    report = scan_and_report_dataset(args.dataset_dir, modes=args.modes, prune=args.prune, output_report=args.report)
    for mode, data in report.items():
        print(f"[{mode}] Total: {data['total']} | Anomalous: {data['anomalous_count']} ({data['anomalous_pct']:.2f}%)")


if __name__ == "__main__":
    main()
