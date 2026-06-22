"""Module 2 & 3: Predictive Modelling (XGBoost/CUDA) and VORP Valuation.

Two regression models:
  A — Expected Minutes (xMin) over the next 4 gameweeks.
  B — Expected FPL Points (xFPL) per gameweek, built from underlying
      xG / xA / xCS metrics rather than raw noisy historic points.

Combined output:
  xP_quality  = (1 - blend) × model_xFPL  +  blend × season_avg_xFPL
  participation = min(1, xMin / 60)        ← full credit at 60+ min
  xP_final    = xP_quality × participation

VORP layer converts projections into scarcity-adjusted value for the
16-manager draft format.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xgboost as xgb

from config import FPLConfig, CONF

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Feature column definitions
# ═══════════════════════════════════════════════════════════════

_XMIN_FEATURES: list[str] = [
    "element_type",
    "xmin_3gw", "xmin_5gw", "xmin_season",
    "start_rate_3gw", "start_rate_5gw", "start_rate_season",
    "was_home",
]

_XP_FEATURES: list[str] = [
    "element_type",
    "is_pen_taker",
    # attacking-role proxy (separates CDMs from AMs/wingers)
    "xgi_rate_3gw", "xgi_rate_5gw", "xgi_rate_season",
    # goal-threat ratio: npxG / xGI — finishers vs creators
    "goal_ratio_3gw", "goal_ratio_5gw", "goal_ratio_season",
    # underlying attacking output
    "npxg_3gw", "npxg_5gw", "npxg_season",
    "xa_3gw", "xa_5gw", "xa_season",
    # defensive output (clean-sheet value for DEF/GKP)
    "xcs_3gw", "xcs_5gw", "xcs_season",
    # bonus tendency
    "bonus_3gw", "bonus_5gw", "bonus_season",
    # fixture difficulty (opponent xG-conceded = how leaky they are)
    "opp_xgc_3gw", "opp_xgc_5gw", "opp_xgc_season",
    # playing-time signal — the model learns the minutes→output
    # relationship internally, so no external participation factor
    "xmin_3gw", "xmin_5gw", "xmin_season",
    "start_rate_3gw", "start_rate_5gw", "start_rate_season",
    "was_home",
]


# ═══════════════════════════════════════════════════════════════
# Training-data builder
# ═══════════════════════════════════════════════════════════════

def build_training_set(
    history: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    horizon: int = 1,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create (X, y) pairs from gameweek history.

    For *horizon* > 1 the target is the mean of the next *horizon* GW values
    for that player (used by the minutes model with horizon=4).
    """
    df = history.copy()

    if horizon == 1:
        df["_target"] = (
            df.groupby("player_id")[target_col]
            .transform(lambda s: s.shift(-1))
        )
    else:
        shifted = []
        for h in range(1, horizon + 1):
            shifted.append(
                df.groupby("player_id")[target_col]
                .transform(lambda s: s.shift(-h))
            )
        stacked = pd.concat(shifted, axis=1)
        df["_target"] = stacked.mean(axis=1)

    df = df.dropna(subset=["_target"] + feature_cols)
    return df[feature_cols], df["_target"]


# ═══════════════════════════════════════════════════════════════
# XGBoost wrappers
# ═══════════════════════════════════════════════════════════════

class _BaseXGBModel:
    """Thin wrapper around XGBRegressor with CUDA defaults."""

    def __init__(self, cfg: FPLConfig = CONF) -> None:
        self.cfg = cfg
        self.model = xgb.XGBRegressor(**cfg.XGB_PARAMS)
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_fraction: float = 0.15) -> None:
        n_eval = max(1, int(len(X) * eval_fraction))
        X_train, X_eval = X.iloc[:-n_eval], X.iloc[-n_eval:]
        y_train, y_eval = y.iloc[:-n_eval], y.iloc[-n_eval:]
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_eval, y_eval)],
            verbose=False,
        )
        self._is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet.")
        return self.model.predict(X)


class MinutesModel(_BaseXGBModel):
    """Model A — predicts average xMin over the next 4 gameweeks."""

    FEATURES = _XMIN_FEATURES
    HORIZON = 4

    def train_from_history(self, history: pd.DataFrame) -> None:
        X, y = build_training_set(
            history, target_col="minutes",
            feature_cols=self.FEATURES, horizon=self.HORIZON,
        )
        log.info("MinutesModel training on %d rows …", len(X))
        self.fit(X, y)

    def project(self, current_features: pd.DataFrame) -> np.ndarray:
        return self.predict(current_features[self.FEATURES]).clip(0, 90)


class PointsModel(_BaseXGBModel):
    """Model B — predicts per-gameweek xFPL directly.

    Trains on ``xfpl`` — the structured expected FPL points per GW built
    from npxG, xA, xCS, appearance, and bonus.  Minutes features are
    included in the feature set so the model learns the minutes→output
    relationship internally.  NO external per-90 normalisation or
    participation factor — the model handles it end-to-end.
    """

    FEATURES = _XP_FEATURES

    def train_from_history(self, history: pd.DataFrame) -> None:
        df = history[history["minutes"] >= 45].copy()
        X, y = build_training_set(
            df, target_col="xfpl",
            feature_cols=self.FEATURES, horizon=1,
        )
        log.info("PointsModel (per-GW xFPL) training on %d rows …", len(X))
        self.fit(X, y)

    def project(self, current_features: pd.DataFrame) -> np.ndarray:
        return self.predict(current_features[self.FEATURES]).clip(0, None)


# ═══════════════════════════════════════════════════════════════
# Projection Engine
# ═══════════════════════════════════════════════════════════════

class ProjectionEngine:
    """Combines MinutesModel + PointsModel with prior blending.

    The PointsModel predicts per-GW xFPL directly (minutes features are
    in the feature set, so the model accounts for playing time internally).
    No external participation factor, no per-90 normalisation.

        xP_model    = PointsModel.predict(snapshot)      per-GW
        xP_quality  = (1-blend) × xP_model + blend × season_xfpl
        xP_final    = clamp(xP_quality, 0, ceiling)

    The MinutesModel still runs for the waiver engine (Module 5) but
    does NOT multiply into xP_final.
    """

    _CEILING: dict[int, float] = {1: 8.0, 2: 10.0, 3: 12.0, 4: 10.0}

    def __init__(self, cfg: FPLConfig = CONF) -> None:
        self.cfg = cfg
        self.minutes_model = MinutesModel(cfg)
        self.points_model = PointsModel(cfg)

    def train(self, history: pd.DataFrame) -> None:
        self.minutes_model.train_from_history(history)
        self.points_model.train_from_history(history)

    def project(self, snapshot: pd.DataFrame) -> pd.DataFrame:
        out = snapshot[["player_id", "element_type"]].copy()

        xmin_raw = self.minutes_model.project(snapshot)
        xp_model = self.points_model.project(snapshot)

        # ── xMin floor overrides for certified nailed-on talismans ──
        # Prevents statistical noise (minor injuries, late-season rest)
        # from tanking draft value of un-benchable assets.
        xmin = xmin_raw.copy()
        if "web_name" in snapshot.columns:
            overrides = self.cfg.XMIN_FLOOR_OVERRIDES
            for name, floor in overrides.items():
                mask = snapshot["web_name"].values == name
                if mask.any():
                    old_val = float(xmin[mask][0])
                    xmin[mask] = np.maximum(xmin[mask], floor)
                    if old_val < floor:
                        log.info("xMin override: %s %.1f → %.1f", name, old_val, floor)

        out["xmin"] = xmin
        out["xp_model"] = xp_model

        # Season prior in per-GW space — anchors elite players
        xfpl_season = snapshot["xfpl_season"].values.astype(float)
        blend = self.cfg.PRIOR_BLEND
        xp_quality = np.where(
            np.isnan(xfpl_season),
            xp_model,
            (1 - blend) * xp_model + blend * xfpl_season,
        )
        out["xp_quality"] = xp_quality

        # Minutes scaling: the model predicts quality-when-appearing,
        # but a player who averages 12 min/GW barely appears at all.
        # xMin / 90 converts per-appearance quality → per-GW expected value.
        participation = xmin / 90.0
        ceiling = out["element_type"].map(self._CEILING).values
        out["xp_final"] = np.clip(xp_quality * participation, 0.0, ceiling)

        return out


# ═══════════════════════════════════════════════════════════════
# Module 3 — VORP Engine
# ═══════════════════════════════════════════════════════════════

class VORPEngine:
    """Value Over Replacement Player for a 16-team draft league.

    Replacement baselines are the Nth-ranked xP_final at each position,
    where N equals the number of starting slots across all 16 teams:
        GKP: 16th   (1 starter × 16)
        DEF: 48th   (3 starters × 16)
        MID: 64th   (4 starters × 16)  ← NOT 80; typical draft leagues
        FWD: 32nd   (2 starters × 16)      start 3-4-3 / 3-5-2 mix
    """

    def __init__(self, cfg: FPLConfig = CONF) -> None:
        self.thresholds = cfg.REPLACEMENT_THRESHOLDS

    def compute(self, projections: pd.DataFrame) -> pd.DataFrame:
        proj = projections.copy()

        baselines: dict[int, float] = {}
        for pos, rank in self.thresholds.items():
            pos_df = proj[proj["element_type"] == pos].sort_values(
                "xp_final", ascending=False
            ).reset_index(drop=True)

            n_available = len(pos_df)
            effective_rank = min(rank, n_available) - 1 if n_available > 0 else 0
            baseline_val = float(pos_df["xp_final"].iloc[effective_rank]) if n_available > 0 else 0.0
            baselines[pos] = baseline_val

            log.info(
                "Pos %d (%3s) replacement baseline: rank %d / %d available → %.2f xP",
                pos,
                {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}.get(pos, "?"),
                rank, n_available, baseline_val,
            )

        proj["replacement_xp"] = proj["element_type"].map(baselines)
        proj["vorp"] = proj["xp_final"] - proj["replacement_xp"]
        return proj


# ═══════════════════════════════════════════════════════════════
# Convenience entry point
# ═══════════════════════════════════════════════════════════════

def latest_player_snapshot(
    history: pd.DataFrame,
    cfg: FPLConfig = CONF,
    recency_gws: int = 8,
) -> pd.DataFrame:
    """Extract the most-recent GW row per player as the feature vector.

    Two purge layers:
      1. Hard exclusion list from config (known ghosts / transferred).
      2. Recency filter — anyone whose last appearance is more than
         *recency_gws* before the season's latest round.
    """
    # Layer 1: hard exclusion
    if "web_name" in history.columns and cfg.EXCLUDED_PLAYERS:
        mask = history["web_name"].isin(cfg.EXCLUDED_PLAYERS)
        if mask.any():
            names = history.loc[mask, "web_name"].unique().tolist()
            log.info("Hard-excluded %d players: %s", len(names), names)
            history = history[~mask]

    season_max = int(history["round"].max())
    last_round = history.groupby("player_id")["round"].max().reset_index()
    last_round.columns = ["player_id", "max_round"]

    # Layer 2: recency filter
    active = last_round[last_round["max_round"] >= season_max - recency_gws]
    dropped = len(last_round) - len(active)
    if dropped:
        log.info("Dropped %d stale players (last appearance > %d GWs ago)",
                 dropped, recency_gws)

    snapshot = history.merge(
        active,
        left_on=["player_id", "round"],
        right_on=["player_id", "max_round"],
    ).drop(columns=["max_round"])
    return snapshot


_POS_LABEL = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

# Players whose FPL-assigned position is known to mismatch their
# real-world role.  Add entries here as the FPL game reclassifies
# players between seasons.  The engine logs a warning but does NOT
# auto-correct because VORP baselines must match FPL's own position
# rules (you draft a "FWD" Bowen against the FWD baseline, not MID).
_POSITION_WATCHLIST: dict[str, str] = {
    "Bowen": "FPL lists as FWD but plays MID/winger in real life",
    "Richarlison": "May flip between FWD/MID across seasons",
}


def _validate_positions(df: pd.DataFrame, label: str) -> None:
    """Hard check that element_type is always in {1,2,3,4}."""
    valid = {1, 2, 3, 4}
    found = set(df["element_type"].dropna().unique().astype(int))
    invalid = found - valid
    if invalid:
        raise ValueError(
            f"[{label}] element_type contains invalid values {invalid}. "
            "Position mapping is broken — VORP baselines will be wrong."
        )
    missing = valid - found
    if missing:
        log.warning("[%s] No players found for positions %s", label, missing)

    for pos in sorted(valid):
        n = int((df["element_type"] == pos).sum())
        log.info("[%s] pos %d (%s): %d players",
                 label, pos, _POS_LABEL[pos], n)

    if "web_name" in df.columns:
        for name, note in _POSITION_WATCHLIST.items():
            rows = df[df["web_name"] == name]
            if len(rows) > 0:
                pos = int(rows["element_type"].iloc[0])
                log.warning(
                    "⚠ POSITION WATCHLIST: %s → pos %d (%s). %s",
                    name, pos, _POS_LABEL.get(pos, "?"), note,
                )


def run_predictions(
    history: pd.DataFrame,
    cfg: FPLConfig = CONF,
) -> pd.DataFrame:
    """Train models on history, project current players, attach VORP."""

    # Position integrity gate — fail fast if mapping is broken
    _validate_positions(history, "history")

    engine = ProjectionEngine(cfg)
    engine.train(history)

    snapshot = latest_player_snapshot(history, cfg=cfg)
    _validate_positions(snapshot, "snapshot")

    projections = engine.project(snapshot)
    _validate_positions(projections, "projections")

    vorp_engine = VORPEngine(cfg)
    return vorp_engine.compute(projections)
