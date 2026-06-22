"""Module 1: Asynchronous FPL data ingestion & feature engineering.

Fetches bootstrap-static, per-player element-summaries, and fixture data
from the official FPL API, then builds rolling statistical features using
pandas (Windows-native).
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import pandas as pd

from config import FPLConfig, CONF

log = logging.getLogger(__name__)

# ── Column subsets kept from each API response ───────────────

_ELEMENT_COLS = [
    "id", "first_name", "second_name", "web_name",
    "team", "element_type", "status",
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
# Local CSV loader (prior seasons)
# ═══════════════════════════════════════════════════════════════

def load_prior_seasons(
    current_players: pd.DataFrame,
    cfg: FPLConfig = CONF,
) -> pd.DataFrame:
    """Load per-GW CSVs from prior season(s) and map to current player IDs.

    Reads ``data/2024-25/players/*/gw.csv``, bridges player IDs via
    (first_name, second_name) matching against the current bootstrap,
    and returns rows with an offset ``round`` column so they sort
    chronologically before the current season.
    """
    data_dir = Path(cfg.LOCAL_DATA_DIR)
    prior_dir = data_dir / "2024-25"

    if not prior_dir.is_dir():
        log.warning("Prior season dir not found: %s", prior_dir)
        return pd.DataFrame()

    # ── build current name → player_id map ───────────────────
    name_to_id: dict[tuple[str, str], int] = {}
    for _, row in current_players.iterrows():
        key = (
            str(row["first_name"]).strip().lower(),
            str(row["second_name"]).strip().lower(),
        )
        name_to_id[key] = int(row["id"])

    # ── build prior element_id → name map via cleaned_players ─
    cleaned_path = prior_dir / "cleaned_players.csv"
    if not cleaned_path.is_file():
        log.warning("No cleaned_players.csv in %s", prior_dir)
        return pd.DataFrame()

    cleaned = pd.read_csv(cleaned_path)
    prev_id_to_name: dict[int, tuple[str, str]] = {}
    for idx, row in cleaned.iterrows():
        eid = idx + 1  # element_id is 1-indexed row order
        prev_id_to_name[eid] = (
            str(row["first_name"]).strip().lower(),
            str(row["second_name"]).strip().lower(),
        )

    # ── read all gw.csv files ────────────────────────────────
    players_dir = prior_dir / "players"
    if not players_dir.is_dir():
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for player_subdir in players_dir.iterdir():
        gw_path = player_subdir / "gw.csv"
        if not gw_path.is_file():
            continue
        try:
            frames.append(pd.read_csv(gw_path))
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    prev = pd.concat(frames, ignore_index=True)
    prev = _coerce_numeric(prev)

    # ── map prior element → current player_id via name bridge ─
    prev["_name_key"] = prev["element"].map(
        lambda eid: prev_id_to_name.get(int(eid), ("", ""))
    )
    prev["player_id"] = prev["_name_key"].map(
        lambda nk: name_to_id.get(nk)
    )
    before = len(prev)
    prev = prev.dropna(subset=["player_id"])
    prev["player_id"] = prev["player_id"].astype(int)
    matched = len(prev)
    log.info(
        "2024-25 local data: %d GW rows loaded, %d matched to current IDs "
        "(%d unmatched / transferred out)",
        before, matched, before - matched,
    )

    # Offset rounds so 2024-25 comes before 2025-26 chronologically
    prev["round"] = prev["round"] - 38

    # Select only the columns the pipeline expects
    keep = [c for c in _HISTORY_COLS if c in prev.columns]
    return prev[keep].copy()


# ═══════════════════════════════════════════════════════════════
# FPL Async Client
# ═══════════════════════════════════════════════════════════════

class FPLClient:
    """Rate-limited async client for the three core FPL endpoints."""

    def __init__(self, cfg: FPLConfig = CONF) -> None:
        self._base = cfg.BASE_URL
        self._sem = asyncio.Semaphore(cfg.API_CONCURRENCY)
        self._delay = cfg.API_REQUEST_DELAY

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

    async def fetch_bootstrap(self, session: aiohttp.ClientSession) -> dict[str, Any]:
        return await self._get_json(session, f"{self._base}/bootstrap-static/")

    async def fetch_player_summary(self, session: aiohttp.ClientSession, player_id: int) -> dict[str, Any]:
        return await self._get_json(session, f"{self._base}/element-summary/{player_id}/")

    async def fetch_fixtures(self, session: aiohttp.ClientSession) -> list[dict[str, Any]]:
        return await self._get_json(session, f"{self._base}/fixtures/")

    async def fetch_all_histories(
        self, session: aiohttp.ClientSession, player_ids: list[int]
    ) -> dict[int, list[dict[str, Any]]]:
        results: dict[int, list[dict[str, Any]]] = {}

        async def _one(pid: int) -> None:
            data = await self.fetch_player_summary(session, pid)
            results[pid] = data.get("history", [])

        await asyncio.gather(*(_one(pid) for pid in player_ids))
        return results

    async def ingest(self) -> tuple[dict[str, Any], dict[int, list[dict]], list[dict]]:
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            bootstrap = await self.fetch_bootstrap(session)
            fixtures = await self.fetch_fixtures(session)

            # Hard filter: only players currently registered in the PL.
            # status: 'a' = available, 'i' = injured, 'd' = doubtful,
            #         'u' = unavailable, 'n' = not in squad / transferred out
            # We keep a/i/d (still at a PL club), drop u/n (gone).
            _ACTIVE_STATUSES = {"a", "i", "d"}
            active_ids = [
                e["id"] for e in bootstrap["elements"]
                if e.get("status", "n") in _ACTIVE_STATUSES
                and (e["minutes"] > 0
                     or e.get("chance_of_playing_next_round") not in (None, 0))
            ]
            dropped = len(bootstrap["elements"]) - len(active_ids)
            log.info(
                "Fetching history for %d active players (%d filtered out as "
                "transferred / unavailable) …", len(active_ids), dropped,
            )
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
    def build_history_frame(histories: dict[int, list[dict[str, Any]]]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for pid, gw_list in histories.items():
            for gw in gw_list:
                rows.append({**gw, "player_id": pid})
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

    # ── helpers ───────────────────────────────────────────────

    def _estimate_npxg(self, history: pd.DataFrame) -> pd.Series:
        pen_scored_est = (
            (history["goals_scored"] - history["expected_goals"])
            .clip(lower=0, upper=2)
        )
        return (history["expected_goals"] - pen_scored_est * self.cfg.PEN_XG_VALUE).clip(lower=0)

    def _team_xgc_table(self, history: pd.DataFrame) -> pd.DataFrame:
        """Per-(team, round) table of xG conceded + rolling averages + xCS."""
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
            team_gw[f"team_xgc_{w}gw"] = (
                team_gw.groupby("team")["xg_conceded"]
                .transform(lambda s: s.rolling(w, min_periods=1).mean())
            )
        team_gw["team_xgc_season"] = (
            team_gw.groupby("team")["xg_conceded"]
            .transform(lambda s: s.expanding(min_periods=1).mean())
        )

        wl = [f"{w}gw" for w in self.cfg.ROLLING_WINDOWS] + ["season"]
        for label in wl:
            team_gw[f"xcs_{label}"] = np.exp(-team_gw[f"team_xgc_{label}"])

        team_gw["xcs_match"] = np.exp(-team_gw["xg_conceded"])
        return team_gw

    def _grouped_rolling(
        self, df: pd.DataFrame, sources: dict[str, str]
    ) -> pd.DataFrame:
        """Add rolling 3gw/5gw/season means for each (source_col → prefix)."""
        for src, prefix in sources.items():
            for w in self.cfg.ROLLING_WINDOWS:
                df[f"{prefix}_{w}gw"] = (
                    df.groupby("player_id")[src]
                    .transform(lambda s: s.rolling(w, min_periods=1).mean())
                )
            df[f"{prefix}_season"] = (
                df.groupby("player_id")[src]
                .transform(lambda s: s.expanding(min_periods=1).mean())
            )
        return df

    # ── main pipeline ────────────────────────────────────────

    def engineer(
        self,
        bootstrap: dict[str, Any],
        histories: dict[int, list[dict]],
        fixtures_raw: list[dict],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        players = self.build_players_frame(bootstrap)
        history = self.build_history_frame(histories)
        fixtures = self.build_fixtures_frame(fixtures_raw)

        # ── Merge local prior-season data ─────────────────────
        prior = load_prior_seasons(players, self.cfg)
        if len(prior) > 0:
            log.info("Augmenting API history (%d rows) with 2024-25 data (%d rows)",
                     len(history), len(prior))
            history = pd.concat([prior, history], ignore_index=True)

        # Join position + team + penalty + name from bootstrap
        history = history.merge(
            players[["id", "web_name", "element_type", "team", "penalties_order"]]
            .rename(columns={"id": "player_id"}),
            on="player_id", how="left",
        )
        history["is_pen_taker"] = np.where(
            history["penalties_order"].fillna(99) <= 1, 1.0, 0.0
        )
        history = history.sort_values(["player_id", "round"]).reset_index(drop=True)

        # ── derived per-GW columns ───────────────────────────
        history["npxg"] = self._estimate_npxg(history)

        # Attacking-role proxy: xGI per minute (separates CDMs from AMs/wingers)
        history["xgi_per_min"] = (
            history["expected_goal_involvements"]
            / history["minutes"].clip(lower=1)
        )

        # Goal-threat ratio: what fraction of xGI is actual goal-scoring?
        # High = clinical finisher (Haaland, Palmer), Low = creative volume
        # without box entry (Enzo, Rice).  FPL rewards goals (4-6 pts)
        # much more than assists (3 pts), so this ratio directly maps to
        # point efficiency within the same xGI volume.
        xgi_safe = history["expected_goal_involvements"].clip(lower=0.01)
        history["goal_threat_ratio"] = (history["npxg"] / xgi_safe).clip(0, 1)

        # ── rolling features ─────────────────────────────────
        history = self._grouped_rolling(history, {
            "npxg":               "npxg",
            "expected_assists":   "xa",
            "minutes":            "xmin",
            "starts":             "start_rate",
            "xgi_per_min":        "xgi_rate",
            "goal_threat_ratio":  "goal_ratio",
            "bonus":              "bonus",
        })

        # ── team-level xGC table ─────────────────────────────
        team_xgc = self._team_xgc_table(history)
        wl = [f"{w}gw" for w in self.cfg.ROLLING_WINDOWS] + ["season"]

        # Merge 1 — own team → xCS (clean-sheet probability for DEF/GKP)
        xcs_cols = [f"xcs_{label}" for label in wl] + ["xcs_match"]
        history = history.merge(
            team_xgc[["team", "round"] + xcs_cols],
            on=["team", "round"], how="left",
        )

        # Merge 2 — opponent → attacking fixture difficulty
        opp_src = [f"team_xgc_{label}" for label in wl]
        opp_df = team_xgc[["team", "round"] + opp_src].copy()
        opp_rename = {"team": "opponent_team"}
        for c in opp_src:
            opp_rename[c] = c.replace("team_xgc_", "opp_xgc_")
        opp_df = opp_df.rename(columns=opp_rename)
        history = history.merge(opp_df, on=["opponent_team", "round"], how="left")

        # ── structured xFPL target (per-GW expected FPL pts) ──
        goal_v = history["element_type"].map(self.cfg.GOAL_PTS).astype(float)
        cs_v = history["element_type"].map(self.cfg.CS_PTS).astype(float)
        appearance = np.where(
            history["minutes"] >= 60, 2.0,
            np.where(history["minutes"] > 0, 1.0, 0.0),
        )
        xfpl_base = (
            history["npxg"] * goal_v
            + history["expected_assists"] * 3.0
            + history["xcs_match"].fillna(0) * cs_v
            + appearance
            + history["bonus"]
        )

        # Talisman coefficient: penalty/set-piece takers generate a
        # structurally higher point ceiling that raw xG/xA understates
        # (penalty xG ≈ 0.76 but the FPL points payoff is 4-6 pts,
        # plus bonus points from penalties scored are near-guaranteed).
        talisman = np.where(
            history["is_pen_taker"] == 1.0,
            self.cfg.TALISMAN_BONUS,
            1.0,
        )
        history["xfpl"] = xfpl_base * talisman

        history = self._grouped_rolling(history, {"xfpl": "xfpl"})

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
    client = FPLClient(cfg)
    bootstrap, histories, fixtures_raw = await client.ingest()
    eng = FeatureEngineer(cfg)
    return eng.engineer(bootstrap, histories, fixtures_raw)
