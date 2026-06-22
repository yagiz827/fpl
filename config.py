"""Central configuration for the FPL Draft Optimization Engine."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FPLConfig:
    """Immutable configuration container for all engine parameters."""

    # ── FPL API ──────────────────────────────────────────────
    BASE_URL: str = "https://fantasy.premierleague.com/api"

    # ── League structure ─────────────────────────────────────
    N_MANAGERS: int = 16
    SQUAD_SIZE: int = 15

    POSITION_NAMES: dict[int, str] = field(default_factory=lambda: {
        1: "GKP", 2: "DEF", 3: "MID", 4: "FWD",
    })

    POSITION_LIMITS: dict[int, int] = field(default_factory=lambda: {
        1: 2, 2: 5, 3: 5, 4: 3,
    })

    # Nth-best starter per position across 16 teams → replacement baseline
    REPLACEMENT_THRESHOLDS: dict[int, int] = field(default_factory=lambda: {
        1: 16, 2: 48, 3: 64, 4: 32,
    })

    # ── Feature engineering ──────────────────────────────────
    ROLLING_WINDOWS: tuple[int, ...] = (3, 5)
    PEN_XG_VALUE: float = 0.76

    # ── XGBoost / CUDA ───────────────────────────────────────
    XGB_PARAMS: dict = field(default_factory=lambda: {
        "tree_method": "hist",
        "device": "cuda",
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    })
    XGB_EARLY_STOPPING: int = 50

    # ── Draft simulation ─────────────────────────────────────
    DRAFT_ROUNDS: int = 15
    MC_SIMULATIONS: int = 10_000
    SURVIVAL_THRESHOLD: float = 0.40
    ADP_SHARPNESS: float = 1.5          # controls pick-probability concentration

    # ── Waiver / streaming ───────────────────────────────────
    WAIVER_HORIZON_GW: int = 3
    XMIN_DROP_THRESHOLD: float = 45.0   # per-GW minutes below which a player is droppable

    # ── Async HTTP ───────────────────────────────────────────
    API_CONCURRENCY: int = 20
    API_REQUEST_DELAY: float = 0.05     # seconds between batched requests


CONF = FPLConfig()
