from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

import ballchasing as bc
from ballchasing.util import from_rfc3339, get_scoreline, TEAMS, get_pid


@dataclass
class BasePlayerInfo:
    id: str = ""
    names: Counter = field(default_factory=Counter)
    first_appearance: datetime = field(default=None)
    latest_appearance: datetime = field(default=None)
    has_liquipedia_page: bool = False
    wins: int = 0
    losses: int = 0

    def update_appearances(self, dt):
        """
        Updates the first and latest appearance dates for the player.
        If first_appearance is None, sets it to the current date.
        If latest_appearance is None or dt is later than the current latest_appearance, updates it.
        """
        if self.first_appearance is None or dt < self.first_appearance:
            self.first_appearance = dt
        if self.latest_appearance is None or dt > self.latest_appearance:
            self.latest_appearance = dt

    @property
    def name(self) -> str:
        """Returns the most common name for this player."""
        if not self.names:
            return self.id  # Fallback to ID if no names are available
        return self.names.most_common(1)[0][0]

    @property
    def games(self) -> int:
        return self.wins + self.losses


@dataclass
class RankedPlayerInfo(BasePlayerInfo):
    ranks: Counter = field(default_factory=Counter)

    @property
    def rank(self) -> str:
        """Returns the most common rank for this player."""
        if not self.ranks:
            return "Unranked"
        return self.ranks.most_common(1)[0][0]

    @property
    def max_rank(self) -> str:
        """Returns the highest rank achieved by this player."""
        return max(self.ranks, key=bc.Rank.ALL.index, default="Unranked")

    @property
    def min_rank(self) -> str:
        """Returns the lowest rank achieved by this player."""
        return min(self.ranks, key=bc.Rank.ALL.index, default="Unranked")


def update_player_infos(pinfos: dict, replay: dict):
    date = from_rfc3339(replay["date"])
    scoreline = get_scoreline(replay)
    for i, team in enumerate(TEAMS):
        won = scoreline[i] > scoreline[1 - i]
        players = replay.get(team, {}).get("players", [])
        for player in players:
            pid = get_pid(player)
            rank = player.get("rank", {}).get("id")
            if rank == bc.Rank.GRAND_CHAMPION_LEGACY:
                rank = bc.Rank.GRAND_CHAMPION_1

            pinfo = pinfos.setdefault(pid, RankedPlayerInfo())
            pinfo.id = pid
            pinfo.names[player.get("name", "")] += 1
            if rank is not None:
                pinfo.ranks[rank] += 1
            if pinfo.first_appearance is None or date < pinfo.first_appearance:
                pinfo.first_appearance = date
            if pinfo.latest_appearance is None or date > pinfo.latest_appearance:
                pinfo.latest_appearance = date
            if player.get("pro", False):
                pinfo.has_liquipedia_page = True
            if won:
                pinfo.wins += 1
            else:
                pinfo.losses += 1
