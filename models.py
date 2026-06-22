"""Module 2 & 3: Predictive Modelling (XGBoost/CUDA) and VORP Valuation.

Two regression models:
  A — Expected Minutes (xMin) over the next 4 gameweeks.
  B — Expected Points per 90 minutes (xP90).

Combined output:  xP_final = xP90 × (xMin / 90)

VORP layer converts raw projections into scarcity-adjusted value for the
16-manager draft format.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xgboost as xgb

from config import FPLConfig, CONF

log = logging.getLogger(__name__)

# ── Feature column definitions ───────────────────────────────

_XMIN_FEATURES: list[str] = [
    "element_type",
    "xmin_3gw", "xmin_5gw", "xmin_season",
    "start_rate_3gw", "start_rate_5gw", "start_rate_season",
    "was_home",
]

_XP90_FEATURES: list[str] = [
    "element_type",
    "npxg_3gw", "npxg_5gw", "npxg_season",
    "xa_3gw", "xa_5gw", "xa_season",
    "xcs_3gw", "xcs_5gw", "xcs_season",
    "pts_3gw", "pts_5gw", "pts_season",
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

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_fraction: float = 0.15,
    ) -> None:
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
    """Model B — predicts xP *per 90 minutes*."""

    FEATURES = _XP90_FEATURES

    def train_from_history(self, history: pd.DataFrame) -> None:
        df = history[history["minutes"] >= 60].copy()
        df["pts_per90"] = df["total_points"] / (df["minutes"] / 90.0)

        X, y = build_training_set(
            df, target_col="pts_per90",
            feature_cols=self.FEATURES, horizon=1,
        )
        log.info("PointsModel training on %d rows …", len(X))
        self.fit(X, y)

    def project(self, current_features: pd.DataFrame) -> np.ndarray:
        return self.predict(current_features[self.FEATURES])


# ═══════════════════════════════════════════════════════════════
# Projection Engine (combines both models)
# ═══════════════════════════════════════════════════════════════

class ProjectionEngine:
    """Produces xP_final = xP90 × (xMin / 90)."""

    def __init__(self, cfg: FPLConfig = CONF) -> None:
        self.minutes_model = MinutesModel(cfg)
        self.points_model = PointsModel(cfg)

    def train(self, history: pd.DataFrame) -> None:
        self.minutes_model.train_from_history(history)
        self.points_model.train_from_history(history)

    def project(self, snapshot: pd.DataFrame) -> pd.DataFrame:
        """Generate projections for all players in *snapshot*.

        *snapshot* must contain the union of MinutesModel.FEATURES and
        PointsModel.FEATURES, plus ``player_id`` and ``element_type``.
        """
        out = snapshot[["player_id", "element_type"]].copy()
        out["xmin"] = self.minutes_model.project(snapshot)
        out["xp90"] = self.points_model.project(snapshot)
        out["xp_final"] = out["xp90"] * (out["xmin"] / 90.0)
        return out


# ═══════════════════════════════════════════════════════════════
# Module 3 — VORP Engine
# ═══════════════════════════════════════════════════════════════

class VORPEngine:
    """Value Over Replacement Player for a 16-team draft league.

    Replacement baselines are the Nth-ranked xP_final at each position,
    where N equals the number of starting slots across 16 teams.
    """

    def __init__(self, cfg: FPLConfig = CONF) -> None:
        self.thresholds = cfg.REPLACEMENT_THRESHOLDS

    def compute(self, projections: pd.DataFrame) -> pd.DataFrame:
        """Add ``replacement_xp`` and ``vorp`` columns."""
        proj = projections.copy()

        baselines: dict[int, float] = {}
        for pos, rank in self.thresholds.items():
            pos_df = proj[proj["element_type"] == pos].sort_values(
                "xp_final", ascending=False
            )
            if len(pos_df) >= rank:
                baselines[pos] = float(pos_df["xp_final"].iloc[rank - 1])
            else:
                baselines[pos] = float(pos_df["xp_final"].min()) if len(pos_df) else 0.0
            log.info("Pos %d replacement baseline (rank %d): %.2f xP",
                     pos, rank, baselines[pos])

        proj["replacement_xp"] = proj["element_type"].map(baselines)
        proj["vorp"] = proj["xp_final"] - proj["replacement_xp"]
        return proj


# ═══════════════════════════════════════════════════════════════
# Convenience entry point
# ═══════════════════════════════════════════════════════════════

def latest_player_snapshot(history: pd.DataFrame) -> pd.DataFrame:
    """Extract the most-recent GW row per player as the feature vector."""
    last_round = history.groupby("player_id")["round"].max().reset_index()
    last_round.columns = ["player_id", "max_round"]
    snapshot = history.merge(
        last_round,
        left_on=["player_id", "round"],
        right_on=["player_id", "max_round"],
    ).drop(columns=["max_round"])
    return snapshot


def run_predictions(
    history: pd.DataFrame,
    cfg: FPLConfig = CONF,
) -> pd.DataFrame:
    """Train models on history, project current players, attach VORP."""
    engine = ProjectionEngine(cfg)
    engine.train(history)

    snapshot = latest_player_snapshot(history)
    projections = engine.project(snapshot)

    vorp_engine = VORPEngine(cfg)
    return vorp_engine.compute(projections)
