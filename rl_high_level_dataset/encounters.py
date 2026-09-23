from dataclasses import dataclass

import ballchasing as bc
from ballchasing.util import get_players, get_pid

from rl_high_level_dataset.player_data import update_player_infos


@dataclass
class EncounterStats:
    pro_count: int = 0
    steam_count: int = 0
    ssl_count: int = 0
    ranked_count: int = 0

    def __add__(self, other):
        if not isinstance(other, EncounterStats):
            return NotImplemented
        return EncounterStats(
            self.pro_count + other.pro_count,
            self.steam_count + other.steam_count,
            self.ssl_count + other.ssl_count,
            self.ranked_count + other.ranked_count
        )

    def __sub__(self, other):
        if not isinstance(other, EncounterStats):
            return NotImplemented
        return EncounterStats(
            self.pro_count - other.pro_count,
            self.steam_count - other.steam_count,
            self.ssl_count - other.ssl_count,
            self.ranked_count - other.ranked_count
        )

    # Beta(0,1) priors ensure no division by 0, probability 0 with no encounters,
    # and strictly increasing probability as the number of encounters increases

    @property
    def prob_pro(self):
        return self.pro_count / (self.steam_count + 1)

    @property
    def prob_ssl(self):
        return self.ssl_count / (self.ranked_count + 1)


def get_encounter_stats(replays):
    encounter_stats = {}
    player_stats = {}
    player_infos = {}
    for replay in replays:
        playlist_id = replay.get("playlist_id")
        pinfos = player_infos.setdefault(playlist_id, {})
        update_player_infos(pinfos, replay)
        if replay.get("playlist_id") != bc.Playlist.RANKED_DOUBLES:
            continue  # Ranked 2s is the main playlist used for practice, so we use that to get fair encounter stats
        players = get_players(replay)
        stats = EncounterStats()
        for i, player in enumerate(players):
            pid = get_pid(player)
            if not pid:
                continue
            own_stats = player_stats.setdefault(pid, EncounterStats())
            platform = (player.get("id") or {}).get("platform")
            if platform == "steam":
                stats.steam_count += 1
                own_stats.steam_count += 1
                if player.get("pro", False):
                    stats.pro_count += 1
                    own_stats.pro_count += 1
            rank = (player.get("rank") or {}).get("id")
            if rank not in (None, bc.Rank.UNRANKED):
                stats.ranked_count += 1
                own_stats.ranked_count += 1
                if rank == bc.Rank.SUPERSONIC_LEGEND:
                    stats.ssl_count += 1
                    own_stats.ssl_count += 1

        for player in players:
            pid = get_pid(player)
            if not pid:
                continue
            encounter_stats[pid] = encounter_stats.get(pid, EncounterStats()) + stats

    return encounter_stats, player_stats, player_infos
