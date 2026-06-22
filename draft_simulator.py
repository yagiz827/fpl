"""Module 4: Monte Carlo Snake-Draft Simulation (CuPy / GPU).

Simulates thousands of 16-team snake drafts in parallel on the GPU.
Uses the Gumbel-max trick for fully-vectorized categorical sampling
across all simulations simultaneously — zero Python-level per-simulation
loops.

Key outputs:
  • Survival probability that a targeted player reaches your next pick.
  • Priority-flag recommendations when survival drops below threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import cupy as cp
import numpy as np

from config import FPLConfig, CONF

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Data containers
# ═══════════════════════════════════════════════════════════════

@dataclass
class PlayerPool:
    """GPU-resident arrays describing the draftable player universe."""

    player_ids: np.ndarray          # (P,) int — original FPL IDs
    names: np.ndarray               # (P,) str
    positions: cp.ndarray           # (P,) int32 — 0-indexed (GKP=0 … FWD=3)
    vorp: cp.ndarray                # (P,) float32 — predicted VORP
    adp_rank: cp.ndarray            # (P,) float32 — 1-indexed ADP rank

    @property
    def n_players(self) -> int:
        return int(self.vorp.shape[0])


@dataclass
class DraftState:
    """Mutable simulation state batched across *n_sims* parallel runs."""

    n_sims: int
    n_managers: int
    n_players: int
    position_limits: cp.ndarray     # (4,) int — max slots per position

    available: cp.ndarray           # (S, P) bool — True if player undrafted
    roster_count: cp.ndarray        # (S, M, 4) int — players drafted per pos

    @classmethod
    def fresh(
        cls,
        n_sims: int,
        n_managers: int,
        n_players: int,
        position_limits: Sequence[int],
    ) -> DraftState:
        return cls(
            n_sims=n_sims,
            n_managers=n_managers,
            n_players=n_players,
            position_limits=cp.array(position_limits, dtype=cp.int32),
            available=cp.ones((n_sims, n_players), dtype=cp.bool_),
            roster_count=cp.zeros((n_sims, n_managers, 4), dtype=cp.int32),
        )


# ═══════════════════════════════════════════════════════════════
# Draft order helpers
# ═══════════════════════════════════════════════════════════════

def snake_order(n_managers: int, n_rounds: int) -> np.ndarray:
    """Generate a full snake-draft pick order.

    Returns shape (n_rounds × n_managers,) array of manager indices.
    """
    order: list[int] = []
    for r in range(n_rounds):
        seq = list(range(n_managers))
        if r % 2 == 1:
            seq.reverse()
        order.extend(seq)
    return np.array(order, dtype=np.int32)


# ═══════════════════════════════════════════════════════════════
# Core simulator
# ═══════════════════════════════════════════════════════════════

class DraftSimulator:
    """GPU-parallel Monte Carlo snake-draft engine."""

    def __init__(self, pool: PlayerPool, cfg: FPLConfig = CONF) -> None:
        self.pool = pool
        self.cfg = cfg

        pos_limits = [cfg.POSITION_LIMITS[p] for p in sorted(cfg.POSITION_LIMITS)]
        self._pos_limits = cp.array(pos_limits, dtype=cp.int32)       # (4,)
        self._order = snake_order(cfg.N_MANAGERS, cfg.DRAFT_ROUNDS)   # (240,)

        self._adp_weights = self._build_adp_weights()                 # (P,)

    # ── ADP → selection probability weights ──────────────────

    def _build_adp_weights(self) -> cp.ndarray:
        """Inverse-power-law weighting from ADP rank.

        Lower ADP rank → higher weight → more likely to be picked.
        """
        alpha = self.cfg.ADP_SHARPNESS
        return 1.0 / (self.pool.adp_rank ** alpha)

    # ── pick probabilities for one draft slot ────────────────

    def _pick_probabilities(
        self,
        state: DraftState,
        manager_idx: int,
    ) -> cp.ndarray:
        """Compute (S, P) selection probabilities for a manager's pick.

        Accounts for:
         1. Player availability (already drafted → 0).
         2. Positional need (roster full at that position → 0).
         3. ADP-weighted preference.
        """
        # positional room: (S, P)
        counts = state.roster_count[:, manager_idx, :]           # (S, 4)
        player_pos = self.pool.positions                         # (P,)
        current = counts[:, player_pos]                          # (S, P)
        limit = self._pos_limits[player_pos]                     # (P,)
        pos_mask = current < limit                               # (S, P)

        probs = (
            self._adp_weights[cp.newaxis, :]                     # (1, P)
            * state.available                                    # (S, P)
            * pos_mask                                           # (S, P)
        )

        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums = cp.where(row_sums > 0, row_sums, 1.0)
        return probs / row_sums

    # ── Gumbel-max categorical sampler ───────────────────────

    @staticmethod
    def _gumbel_sample(probs: cp.ndarray) -> cp.ndarray:
        """Sample one index per row via the Gumbel-max trick.

        Parameters
        ----------
        probs : (S, P) float32 — row-normalized probabilities.

        Returns
        -------
        (S,) int32 — sampled player index per simulation.
        """
        u = cp.random.uniform(low=1e-20, high=1.0, size=probs.shape)
        gumbel = -cp.log(-cp.log(u))
        log_probs = cp.where(probs > 0, cp.log(probs + 1e-30), -1e30)
        return cp.argmax(log_probs + gumbel, axis=1).astype(cp.int32)

    # ── apply a pick to state ────────────────────────────────

    def _apply_pick(
        self,
        state: DraftState,
        manager_idx: int,
        selected: cp.ndarray,
    ) -> None:
        """Update availability and roster counts in-place."""
        sim_idx = cp.arange(state.n_sims)
        state.available[sim_idx, selected] = False

        sel_pos = self.pool.positions[selected]                  # (S,)
        for p in range(4):
            mask = sel_pos == p
            state.roster_count[mask, manager_idx, p] += 1

    # ── full simulation ──────────────────────────────────────

    def simulate(
        self,
        my_manager: int = 0,
        already_drafted: dict[int, list[int]] | None = None,
        start_pick: int = 0,
        n_sims: int | None = None,
    ) -> tuple[cp.ndarray, DraftState]:
        """Run the Monte Carlo simulation from *start_pick* onward.

        Parameters
        ----------
        my_manager : int
            Your 0-indexed seat at the draft table.
        already_drafted : dict mapping manager_idx → list of player pool
            indices already on their roster (for mid-draft entry).
        start_pick : int
            Index into the full pick order to resume from.
        n_sims : int | None
            Override default simulation count.

        Returns
        -------
        my_picks : (S, R) int32 — player indices *you* selected per sim.
        final_state : DraftState
        """
        n = n_sims or self.cfg.MC_SIMULATIONS
        P = self.pool.n_players
        M = self.cfg.N_MANAGERS

        state = DraftState.fresh(n, M, P, self._pos_limits.get().tolist())

        if already_drafted:
            for mgr, indices in already_drafted.items():
                for idx in indices:
                    state.available[:, idx] = False
                    pos = int(self.pool.positions[idx])
                    state.roster_count[:, mgr, pos] += 1

        my_picks_list: list[cp.ndarray] = []

        for pick_seq in range(start_pick, len(self._order)):
            mgr = int(self._order[pick_seq])

            if mgr == my_manager:
                # "Your" pick — take highest available VORP (deterministic
                # from your perspective; stochasticity is in opponents).
                masked_vorp = cp.where(
                    state.available, self.pool.vorp[cp.newaxis, :], -1e30
                )

                counts = state.roster_count[:, mgr, :]
                player_pos = self.pool.positions
                current = counts[:, player_pos]
                limit = self._pos_limits[player_pos]
                pos_ok = current < limit
                masked_vorp = cp.where(pos_ok, masked_vorp, -1e30)

                selected = cp.argmax(masked_vorp, axis=1).astype(cp.int32)
                my_picks_list.append(selected)
            else:
                probs = self._pick_probabilities(state, mgr)
                selected = self._gumbel_sample(probs)

            self._apply_pick(state, mgr, selected)

        my_picks = cp.stack(my_picks_list, axis=1) if my_picks_list else cp.empty(
            (n, 0), dtype=cp.int32
        )
        return my_picks, state

    # ── survival probability ─────────────────────────────────

    def survival_probability(
        self,
        target_player_idx: int,
        my_manager: int,
        current_pick: int,
        already_drafted: dict[int, list[int]] | None = None,
        n_sims: int | None = None,
    ) -> float:
        """Probability that *target_player_idx* is still on the board
        when *my_manager*'s next turn arrives after *current_pick*.

        Uses a dedicated, shorter simulation that terminates right after
        the next turn for efficiency.
        """
        n = n_sims or self.cfg.MC_SIMULATIONS
        P = self.pool.n_players
        M = self.cfg.N_MANAGERS

        next_turn: int | None = None
        for i in range(current_pick, len(self._order)):
            if int(self._order[i]) == my_manager:
                next_turn = i
                break

        if next_turn is None:
            return 0.0

        state = DraftState.fresh(n, M, P, self._pos_limits.get().tolist())
        if already_drafted:
            for mgr, indices in already_drafted.items():
                for idx in indices:
                    state.available[:, idx] = False
                    pos = int(self.pool.positions[idx])
                    state.roster_count[:, mgr, pos] += 1

        for pick_seq in range(current_pick, next_turn):
            mgr = int(self._order[pick_seq])
            if mgr == my_manager:
                continue
            probs = self._pick_probabilities(state, mgr)
            selected = self._gumbel_sample(probs)
            self._apply_pick(state, mgr, selected)

        survived = state.available[:, target_player_idx]
        return float(survived.mean())

    # ── recommendation layer ─────────────────────────────────

    def recommend(
        self,
        my_manager: int,
        current_pick: int,
        watchlist: list[int],
        already_drafted: dict[int, list[int]] | None = None,
        n_sims: int | None = None,
    ) -> list[dict]:
        """Flag watchlist players whose survival probability is below
        the configured threshold.

        Returns a sorted list of dicts with fields:
            player_idx, name, vorp, survival_prob, priority
        """
        results: list[dict] = []
        for pidx in watchlist:
            sp = self.survival_probability(
                pidx, my_manager, current_pick, already_drafted, n_sims
            )
            results.append({
                "player_idx": pidx,
                "name": str(self.pool.names[pidx]),
                "vorp": float(self.pool.vorp[pidx]),
                "survival_prob": sp,
                "priority": sp < self.cfg.SURVIVAL_THRESHOLD,
            })

        results.sort(key=lambda r: r["survival_prob"])
        return results


# ═══════════════════════════════════════════════════════════════
# Builder — construct PlayerPool from a projections DataFrame
# ═══════════════════════════════════════════════════════════════

def build_player_pool(
    projections: "pd.DataFrame",
    names_map: dict[int, str] | None = None,
) -> PlayerPool:
    """Materialise a PlayerPool from a projections frame (output of models.py).

    The frame must contain: player_id, element_type, vorp.
    ADP rank defaults to VORP-descending rank when no external ADP data
    is supplied.
    """
    proj_pd = projections.sort_values("vorp", ascending=False).reset_index(drop=True)

    ids = proj_pd["player_id"].values
    positions = proj_pd["element_type"].values.astype(np.int32) - 1   # 0-index
    vorp = proj_pd["vorp"].values.astype(np.float32)
    adp_rank = np.arange(1, len(proj_pd) + 1, dtype=np.float32)

    if names_map:
        names = np.array([names_map.get(int(i), str(i)) for i in ids])
    else:
        names = np.array([str(i) for i in ids])

    return PlayerPool(
        player_ids=ids,
        names=names,
        positions=cp.asarray(positions),
        vorp=cp.asarray(vorp),
        adp_rank=cp.asarray(adp_rank),
    )
