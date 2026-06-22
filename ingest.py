"""Module 1: Asynchronous FPL data ingestion & feature engineering.

Fetches bootstrap-static, per-player element-summaries, and fixture data
from the official FPL API, then builds rolling statistical features using
pandas (Windows-native).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp
import numpy as np
import pandas as pd

from config import FPLConfig, CONF

log = logging.getLogger(__name__)

# ── Column subsets kept from each API response ───────────────

_ELEMENT_COLS = [
    "id", "web_name", "team", "element_type", "status",
    "chance_of_playing_next_round", "penalties_order",
    "minutes", "starts", "expected_goals", "expected_assists",
    "expected_goal_involvements", "expected_goals_conceded",
    "goals_scored", "assists", "clean_sheets",
    "total_points", "bonus", "penalties_missed",
]

_HISTORY_COLS = [
    "player_id", "round", "minutes", "starts",
    "goals_scored", "assists", "clean_sheets",
    "expected_goals", "expected_assists",
    "expected_goal_involvements", "expected_goals_conceded",
    "total_points", "bonus", "penalties_missed",
    "opponent_team", "was_home",
]

_FIXTURE_COLS = [
    "id", "event", "team_h", "team_a",
    "team_h_difficulty", "team_a_difficulty", "finished",
]

# Columns the FPL API returns as strings that must be numeric
_NUMERIC_COLS = {
    "minutes", "starts", "goals_scored", "assists", "clean_sheets",
    "expected_goals", "expected_assists",
    "expected_goal_involvements", "expected_goals_conceded",
    "total_points", "bonus", "penalties_missed", "penalties_order",
    "chance_of_playing_next_round",
    "team_h_difficulty", "team_a_difficulty",
}


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Cast known-numeric columns that the API may return as strings."""
    for col in df.columns:
        if col in _NUMERIC_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ═══════════════════════════════════════════════════════════════
# FPL Async Client
# ═══════════════════════════════════════════════════════════════

class FPLClient:
    """Rate-limited async client for the three core FPL endpoints."""

    def __init__(self, cfg: FPLConfig = CONF) -> None:
        self._base = cfg.BASE_URL
        self._sem = asyncio.Semaphore(cfg.API_CONCURRENCY)
        self._delay = cfg.API_REQUEST_DELAY

    # ── low-level GET ────────────────────────────────────────

    async def _get_json(
        self, session: aiohttp.ClientSession, url: str, retries: int = 3
    ) -> Any:
        for attempt in range(1, retries + 1):
            try:
                async with self._sem:
                    async with session.get(url) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                    await asyncio.sleep(self._delay)
                    return data
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                wait = 2.0 ** attempt
                log.warning("attempt %d for %s failed (%s), retrying in %.1fs",
                            attempt, url, exc, wait)
                await asyncio.sleep(wait)
        raise RuntimeError(f"Failed to fetch {url} after {retries} retries")

    # ── endpoint wrappers ────────────────────────────────────

    async def fetch_bootstrap(
        self, session: aiohttp.ClientSession
    ) -> dict[str, Any]:
        return await self._get_json(session, f"{self._base}/bootstrap-static/")

    async def fetch_player_summary(
        self, session: aiohttp.ClientSession, player_id: int
    ) -> dict[str, Any]:
        url = f"{self._base}/element-summary/{player_id}/"
        return await self._get_json(session, url)

    async def fetch_fixtures(
        self, session: aiohttp.ClientSession
    ) -> list[dict[str, Any]]:
        return await self._get_json(session, f"{self._base}/fixtures/")

    # ── bulk player history ──────────────────────────────────

    async def fetch_all_histories(
        self, session: aiohttp.ClientSession, player_ids: list[int]
    ) -> dict[int, list[dict[str, Any]]]:
        results: dict[int, list[dict[str, Any]]] = {}

        async def _one(pid: int) -> None:
            data = await self.fetch_player_summary(session, pid)
            results[pid] = data.get("history", [])

        await asyncio.gather(*(_one(pid) for pid in player_ids))
        return results

    # ── top-level ingest ─────────────────────────────────────

    async def ingest(
        self,
    ) -> tuple[dict[str, Any], dict[int, list[dict]], list[dict]]:
        """Fetch bootstrap, all active-player histories, and fixtures."""
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            bootstrap = await self.fetch_bootstrap(session)
            fixtures = await self.fetch_fixtures(session)

            active_ids = [
                e["id"]
                for e in bootstrap["elements"]
                if e["minutes"] > 0
                or e.get("chance_of_playing_next_round") not in (None, 0)
            ]
            log.info("Fetching history for %d active players …", len(active_ids))
            histories = await self.fetch_all_histories(session, active_ids)

        return bootstrap, histories, fixtures


# ═══════════════════════════════════════════════════════════════
# Feature Engineer (pandas / NumPy)
# ═══════════════════════════════════════════════════════════════

class FeatureEngineer:
    """Builds rolling statistical features on CPU via pandas."""

    def __init__(self, cfg: FPLConfig = CONF) -> None:
        self.cfg = cfg

    # ── dataframe builders ───────────────────────────────────

    @staticmethod
    def build_players_frame(bootstrap: dict[str, Any]) -> pd.DataFrame:
        pdf = pd.DataFrame(bootstrap["elements"])
        for c in _ELEMENT_COLS:
            if c not in pdf.columns:
                pdf[c] = np.nan
        return _coerce_numeric(pdf[_ELEMENT_COLS].copy())

    @staticmethod
    def build_history_frame(
        histories: dict[int, list[dict[str, Any]]]
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for pid, gw_list in histories.items():
            for gw in gw_list:
                row = {**gw, "player_id": pid}
                rows.append(row)
        if not rows:
            raise ValueError("No gameweek history available — is the season underway?")
        pdf = pd.DataFrame(rows)
        for c in _HISTORY_COLS:
            if c not in pdf.columns:
                pdf[c] = np.nan
        return _coerce_numeric(pdf[_HISTORY_COLS].copy())

    @staticmethod
    def build_fixtures_frame(fixtures: list[dict[str, Any]]) -> pd.DataFrame:
        pdf = pd.DataFrame(fixtures)
        for c in _FIXTURE_COLS:
            if c not in pdf.columns:
                pdf[c] = np.nan
        return _coerce_numeric(pdf[_FIXTURE_COLS].copy())

    # ── non-penalty xG ───────────────────────────────────────

    def _estimate_npxg(self, history: pd.DataFrame) -> pd.Series:
        """npxG ≈ xG − (penalties_scored × 0.76).

        Penalties scored per GW is unobservable directly; we approximate it
        as max(0, goals − xG) clipped to at most 2.  This is a conservative
        heuristic — for higher accuracy, supplement with Understat/FBref data.
        """
        pen_scored_est = (
            (history["goals_scored"] - history["expected_goals"])
            .clip(lower=0, upper=2)
        )
        return (history["expected_goals"] - pen_scored_est * self.cfg.PEN_XG_VALUE).clip(lower=0)

    # ── team-level xG-conceded rolling ───────────────────────

    def _team_xgc_rolling(self, history: pd.DataFrame) -> pd.DataFrame:
        """Per-team rolling non-penalty xG conceded (opponent attacking strength)."""
        team_gw = (
            history
            .groupby(["opponent_team", "round"], sort=False)
            .agg({"expected_goals": "sum", "goals_scored": "sum"})
            .reset_index()
            .rename(columns={
                "opponent_team": "team",
                "expected_goals": "xg_conceded",
                "goals_scored": "goals_conceded",
            })
        )
        team_gw = team_gw.sort_values(["team", "round"]).reset_index(drop=True)

        for w in self.cfg.ROLLING_WINDOWS:
            col = f"team_xgc_{w}gw"
            team_gw[col] = (
                team_gw.groupby("team")["xg_conceded"]
                .transform(lambda s: s.rolling(w, min_periods=1).mean())
            )

        team_gw["team_xgc_season"] = (
            team_gw.groupby("team")["xg_conceded"]
            .transform(lambda s: s.expanding(min_periods=1).mean())
        )
        return team_gw

    # ── expected clean-sheet probability ─────────────────────

    @staticmethod
    def _xcs_from_xgc(xgc_series: pd.Series) -> pd.Series:
        """P(clean sheet) ≈ e^(−opponent_xG) under a Poisson model."""
        return np.exp(-xgc_series)

    # ── main pipeline ────────────────────────────────────────

    def engineer(
        self,
        bootstrap: dict[str, Any],
        histories: dict[int, list[dict]],
        fixtures_raw: list[dict],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run full feature-engineering pipeline.

        Returns
        -------
        players : pd.DataFrame   — static player attributes + season totals
        history : pd.DataFrame   — per-GW rows with rolling features attached
        fixtures : pd.DataFrame  — future fixture metadata
        """
        players = self.build_players_frame(bootstrap)
        history = self.build_history_frame(histories)
        fixtures = self.build_fixtures_frame(fixtures_raw)

        history = history.sort_values(["player_id", "round"]).reset_index(drop=True)

        # non-penalty xG per gameweek
        history["npxg"] = self._estimate_npxg(history)

        # ── rolling features per player ──────────────────────
        roll_cols = {
            "npxg":             "npxg",
            "expected_assists": "xa",
            "minutes":          "xmin",
            "starts":           "start_rate",
            "total_points":     "pts",
        }

        for src, prefix in roll_cols.items():
            for w in self.cfg.ROLLING_WINDOWS:
                dst = f"{prefix}_{w}gw"
                history[dst] = (
                    history.groupby("player_id")[src]
                    .transform(lambda s: s.rolling(w, min_periods=1).mean())
                )

            history[f"{prefix}_season"] = (
                history.groupby("player_id")[src]
                .transform(lambda s: s.expanding(min_periods=1).mean())
            )

        # ── team-level xGC rolling → xCS ─────────────────────
        team_xgc = self._team_xgc_rolling(history)

        for w_label in [f"{w}gw" for w in self.cfg.ROLLING_WINDOWS] + ["season"]:
            xgc_col = f"team_xgc_{w_label}"
            xcs_col = f"xcs_{w_label}"
            team_xgc[xcs_col] = self._xcs_from_xgc(team_xgc[xgc_col])

        merge_cols = (
            ["team", "round"]
            + [f"team_xgc_{w_label}" for w_label in
               [f"{w}gw" for w in self.cfg.ROLLING_WINDOWS] + ["season"]]
            + [f"xcs_{w_label}" for w_label in
               [f"{w}gw" for w in self.cfg.ROLLING_WINDOWS] + ["season"]]
        )

        history = history.merge(
            team_xgc[merge_cols],
            left_on=["opponent_team", "round"],
            right_on=["team", "round"],
            how="left",
        ).drop(columns=["team"], errors="ignore")

        log.info(
            "Feature engineering complete: %d player-GW rows, %d columns",
            len(history), len(history.columns),
        )
        return players, history, fixtures


# ═══════════════════════════════════════════════════════════════
# Convenience entry point
# ═══════════════════════════════════════════════════════════════

async def run_ingestion(
    cfg: FPLConfig = CONF,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """End-to-end: fetch → parse → feature-engineer, return DataFrames."""
    client = FPLClient(cfg)
    bootstrap, histories, fixtures_raw = await client.ingest()
    eng = FeatureEngineer(cfg)
    return eng.engineer(bootstrap, histories, fixtures_raw)
