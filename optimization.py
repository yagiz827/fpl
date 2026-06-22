"""Module 5: Weekly Waiver & Free-Agency Streaming Engine.

Evaluates the free-agent pool over a rolling multi-gameweek horizon,
identifies droppable roster assets (degraded xMin), and recommends
streaming pickups — prioritising defensive assets with favourable
immediate fixture pairings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from config import FPLConfig, CONF

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Fixture-difficulty helpers
# ═══════════════════════════════════════════════════════════════

def upcoming_fixture_difficulty(
    fixtures: pd.DataFrame,
    current_gw: int,
    horizon: int,
) -> pd.DataFrame:
    """Per-team average fixture difficulty over the next *horizon* GWs.

    Returns a DataFrame with columns: team, avg_difficulty, n_fixtures.
    """
    upcoming = fixtures[
        (fixtures["event"] > current_gw)
        & (fixtures["event"] <= current_gw + horizon)
        & (~fixtures["finished"])
    ]

    home = upcoming[["team_h", "team_h_difficulty"]].rename(
        columns={"team_h": "team", "team_h_difficulty": "difficulty"}
    )
    away = upcoming[["team_a", "team_a_difficulty"]].rename(
        columns={"team_a": "team", "team_a_difficulty": "difficulty"}
    )
    combined = pd.concat([home, away], ignore_index=True)
    agg = (
        combined.groupby("team")["difficulty"]
        .agg(["mean", "count"])
        .reset_index()
    )
    agg.columns = ["team", "avg_difficulty", "n_fixtures"]
    return agg


def fixture_attractiveness(avg_difficulty: pd.Series) -> pd.Series:
    """Invert difficulty (FDR 1-5) into an attractiveness score in (0,1].

    Lower difficulty -> higher attractiveness.
    """
    score = 1.0 / avg_difficulty
    return score / score.max()


# ═══════════════════════════════════════════════════════════════
# Drop-candidate identification
# ═══════════════════════════════════════════════════════════════

def identify_drop_candidates(
    roster: pd.DataFrame,
    projections: pd.DataFrame,
    cfg: FPLConfig = CONF,
) -> pd.DataFrame:
    """Flag roster players whose projected xMin has fallen below
    the configured threshold — indicating minutes risk.

    Parameters
    ----------
    roster : pd.DataFrame
        Must contain ``player_id``.
    projections : pd.DataFrame
        Must contain ``player_id``, ``xmin``, ``xp_final``, ``vorp``.

    Returns
    -------
    pd.DataFrame — roster subset with projections joined, sorted by
    ascending xMin (worst first).
    """
    merged = roster.merge(projections, on="player_id", how="left")
    drops = merged[merged["xmin"] < cfg.XMIN_DROP_THRESHOLD]
    return drops.sort_values("xmin", ascending=True).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
# Free-agent evaluation
# ═══════════════════════════════════════════════════════════════

def evaluate_free_agents(
    projections: pd.DataFrame,
    rostered_ids: Sequence[int],
    fixtures: pd.DataFrame,
    current_gw: int,
    cfg: FPLConfig = CONF,
) -> pd.DataFrame:
    """Score every unrostered player over the waiver horizon.

    The composite waiver score blends projected value (VORP) with
    fixture attractiveness:

        waiver_score = vorp_norm * 0.6 + fixture_attract * 0.4

    Parameters
    ----------
    projections : pd.DataFrame
        Full player projections (player_id, element_type, xmin, xp_final, vorp,
        and a ``team`` column for fixture lookup).
    rostered_ids : sequence of int
        Player IDs currently on any manager's roster.
    fixtures : pd.DataFrame
        Fixture data from ingest.py.
    current_gw : int
        The most recently completed gameweek number.

    Returns
    -------
    pd.DataFrame — free agents ranked by waiver_score descending.
    """
    rostered_set = set(int(x) for x in rostered_ids)
    mask = ~projections["player_id"].isin(list(rostered_set))
    fa = projections[mask].copy()

    if len(fa) == 0:
        return fa

    fd = upcoming_fixture_difficulty(fixtures, current_gw, cfg.WAIVER_HORIZON_GW)
    fa = fa.merge(fd, on="team", how="left")
    fa["fixture_attract"] = fixture_attractiveness(fa["avg_difficulty"])

    vorp_min = float(fa["vorp"].min())
    vorp_max = float(fa["vorp"].max())
    vorp_range = max(vorp_max - vorp_min, 1e-6)
    fa["vorp_norm"] = (fa["vorp"] - vorp_min) / vorp_range

    fa["waiver_score"] = fa["vorp_norm"] * 0.6 + fa["fixture_attract"] * 0.4

    return fa.sort_values("waiver_score", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
# Streaming-defender finder
# ═══════════════════════════════════════════════════════════════

def find_streaming_defenders(
    free_agents: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """From the scored free-agent pool, extract top *top_n* defenders/GKPs
    with the best fixture attractiveness — the classic streaming strategy.

    Streaming prioritises fixture quality over raw talent because these
    assets will be rotated frequently.
    """
    streamable = free_agents[free_agents["element_type"].isin([1, 2])].copy()
    if len(streamable) == 0:
        return streamable

    streamable["stream_score"] = (
        streamable["fixture_attract"] * 0.7
        + streamable["vorp_norm"] * 0.3
    )
    return (
        streamable
        .sort_values("stream_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ═══════════════════════════════════════════════════════════════
# Unified Waiver Engine
# ═══════════════════════════════════════════════════════════════

@dataclass
class WaiverMove:
    """A single recommended add/drop transaction."""
    drop_player_id: int
    drop_name: str
    drop_xmin: float
    add_player_id: int
    add_name: str
    add_waiver_score: float
    add_position: int
    net_vorp_gain: float


class WaiverEngine:
    """Top-level waiver optimiser that pairs drops with adds."""

    def __init__(self, cfg: FPLConfig = CONF) -> None:
        self.cfg = cfg

    def recommend(
        self,
        roster: pd.DataFrame,
        projections: pd.DataFrame,
        all_rostered_ids: Sequence[int],
        fixtures: pd.DataFrame,
        current_gw: int,
        names_map: dict[int, str] | None = None,
    ) -> list[WaiverMove]:
        """Generate ranked waiver recommendations.

        Steps:
          1. Identify droppable roster assets (low xMin).
          2. Score the free-agent pool over the waiver horizon.
          3. Greedily match each drop to the best available same-position add
             that yields a positive VORP gain.
        """
        drops = identify_drop_candidates(roster, projections, self.cfg)
        fa = evaluate_free_agents(
            projections, all_rostered_ids, fixtures, current_gw, self.cfg
        )

        if len(drops) == 0 or len(fa) == 0:
            log.info("No actionable waiver moves found.")
            return []

        nm = names_map or {}

        used_adds: set[int] = set()
        moves: list[WaiverMove] = []

        for _, drop_row in drops.iterrows():
            drop_pos = int(drop_row["element_type"])
            drop_vorp = float(drop_row.get("vorp", 0))

            candidates = fa[
                (fa["element_type"] == drop_pos)
                & (~fa["player_id"].isin(used_adds))
            ]
            if candidates.empty:
                continue

            best = candidates.iloc[0]
            add_vorp = float(best.get("vorp", 0))
            gain = add_vorp - drop_vorp

            if gain <= 0:
                continue

            used_adds.add(int(best["player_id"]))
            moves.append(WaiverMove(
                drop_player_id=int(drop_row["player_id"]),
                drop_name=nm.get(int(drop_row["player_id"]),
                                 str(int(drop_row["player_id"]))),
                drop_xmin=float(drop_row["xmin"]),
                add_player_id=int(best["player_id"]),
                add_name=nm.get(int(best["player_id"]),
                                str(int(best["player_id"]))),
                add_waiver_score=float(best["waiver_score"]),
                add_position=int(best["element_type"]),
                net_vorp_gain=gain,
            ))

        moves.sort(key=lambda m: m.net_vorp_gain, reverse=True)
        log.info("Recommended %d waiver moves.", len(moves))
        return moves
