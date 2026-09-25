# High-Level Rocket League Replay Dataset

A pipeline and curated dataset of high-tier competitive Rocket League matches sourced from [ballchasing.com](https://ballchasing.com), designed for imitation learning, trajectory forecasting, and offline reinforcement learning.

The complete dataset is hosted on Kaggle:  
👉 **[High-Level Rocket League Replay Dataset on Kaggle](https://www.kaggle.com/datasets/rolvarild/high-level-rocket-league-replay-dataset)**

Unlike raw public replay dumps, this dataset undergoes graph-based skill verification, anti-cheat screening, behavioral rate validation, and statistical distribution hypothesis testing to eliminate non-serious games, content-creator handicaps, mutators, and bot matches.

---

## Dataset Breakdown

Modes are balanced by **active gameplay clock time** (excluding kickoff countdowns and post-goal replays), greedily selecting matches with the highest average player skill scores.

| Mode | Matches | Gameplay Duration | Player Trajectory Time | Players / Match |
| :---: | :---: | :---: | :---: | :---: |
| **1v1** | 48,435 | 3,788 hours | 7,577 hours | 2 |
| **2v2** | 49,658 | 3,788 hours | 15,154 hours | 4 |
| **3v3** | 42,235 | 3,788 hours | 22,730 hours | 6 |
| **Total** | **140,328** | **11,365 hours** | **45,460 hours** | — |

---

## Collection Strategy

1. **Ranked Discovery**: Collect GC+ ranked replays across modern free-to-play seasons (S5 to S23).
2. **Skill Scoring**: In ranked 2v2, score players based on how often they encounter verified Supersonic Legends (SSL) and Liquipedia-tagged professionals.
3. **Private & Tournament Scrims**: Include private and off-ladder replays featuring players with qualifying skill scores.
4. **Anti-Cheat & Bot Pruning**: Filter against known RLGym/injection botting accounts (Nexto, Element) and banlists (`cheaters.txt` and `whosbotting`).
5. **Score-Prioritized Balancing**: Pick top replays in descending order of average player score until 1v1, 2v2, and 3v3 match durations are balanced.

---

## Quality & Anomaly Filters

To ensure the dataset represents genuine, high-effort competitive play, every match is audited for behavioral and statistical anomalies:

| Filter | Method / Metric | What It Catches |
| :--- | :--- | :--- |
| **No-Powerslide Floor** | Powerslides < 1.5 / min | Content-creator handicap challenges (*"No powerslide to SSL"*, keyboard-only) |
| **Activity Floors** | BCPM, BPM, and average speed percentiles | AFK players, controller disconnects, boost-camping, and griefing |
| **Positioning Floors** | Ball distance and time ahead of the ball | Extended goal-sitting, trolling, and rule-1 lockouts |
| **Goal Disparity** | Skellam test (p < 10^-4) | Early blowout forfeits (e.g. 7-0 in 2.4 min), win-trading, and intentional throwing |
| **Goal Frequency** | Poisson test (p < 10^-4) | Kickoff-goal spam and mutator matches (unlimited boost, low gravity) |
| **Overtime Length** | Exponential test (p < 10^-5) | Infinite-length private lobbies (> 15 min OT) without high save counts |
| **Freestyle 1v1** | Joint positional / airtime pattern | Turn-based aerial practice sessions from opposite goal lines |
| **RLCS Dedicated Servers** | Server header inspection (`RLCS*`) | Protects tactical pro variance (demo meta, 3rd-man roles) while screening crashes |

---

## Installation

This project uses [uv](https://github.com/astral-sh/uv) (or standard `pip` with Python >= 3.13):

```bash
git clone https://github.com/Rolv-Arild/rl-high-level-dataset.git
cd rl-high-level-dataset

# Install dependencies using uv
uv sync
```

Alternatively, with `pip`:

```bash
pip install .
```

---

## Usage

### 1. Full Dataset Pipeline

Set your `BALLCHASING_API_KEY` environment variable:

```bash
export BALLCHASING_API_KEY="your_api_key_here"
```

Run the pipeline:

```bash
python -m rl_high_level_dataset.main \
    --base-dir ./out/high_level \
    --out-path /path/to/destination
```

Key arguments:
* `--base-dir`: Directory for caching API responses, scores, and deep replay metadata.
* `--out-path`: Destination directory where `.replay` files and `metadata.json` will be stored.
* `--disable-anomaly-filter`: Disable statistical and behavioral anomaly filtering if uncurated raw matches are desired.

### 2. Standalone Anomaly Audit & Pruning CLI

You can audit or clean existing replay datasets directly without re-running the main pipeline:

```bash
# Dry-run audit:
python -m rl_high_level_dataset.anomaly --dataset-dir /path/to/dataset

# Audit and prune anomalous .replay files from disk while updating metadata.json:
python -m rl_high_level_dataset.anomaly --dataset-dir /path/to/dataset --prune --report report.json
```

---

## Output Metadata Format

Each mode folder contains raw `.replay` files alongside a line-delimited `metadata.json`. Each entry is a JSON object containing:

```json
{
  "id": "replay-uuid",
  "score": 0.852,
  "gameplay_duration": 298.5,
  "players": ["player_id_1", "player_id_2"],
  "data": {
    "title": "Match title",
    "date": "2024-05-10T15:46:00Z",
    "server": { "name": "USE123-Standard", "region": "USE" },
    "blue": { "players": [...] },
    "orange": { "players": [...] }
  }
}
```
