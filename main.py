"""FPL Draft Optimisation Engine — CLI entry point.

Usage examples
--------------
Full pipeline (ingest -> train -> project -> VORP):
    python main.py pipeline

Draft simulation from pick position 3:
    python main.py draft --pick-position 3

Weekly waiver analysis for gameweek 12:
    python main.py waiver --current-gw 12 --roster-ids 10,44,87,102,...
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

import cupy as cp
import pandas as pd

from config import CONF, FPLConfig
from ingest import run_ingestion
from models import run_predictions, latest_player_snapshot
from draft_simulator import DraftSimulator, build_player_pool
from optimization import (
    WaiverEngine,
    evaluate_free_agents,
    find_streaming_defenders,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-22s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fpl_engine")


# ═══════════════════════════════════════════════════════════════
# Sub-commands
# ═══════════════════════════════════════════════════════════════

async def cmd_pipeline(cfg: FPLConfig) -> pd.DataFrame:
    """Ingest -> feature-engineer -> train models -> project -> VORP."""
    log.info("▸ Starting full pipeline")

    players, history, fixtures = await run_ingestion(cfg)

    names_map: dict[int, str] = dict(
        zip(
            players["id"].tolist(),
            players["web_name"].tolist(),
        )
    )

    projections = run_predictions(history, cfg)

    projections = projections.merge(
        players[["id", "team", "web_name"]].rename(columns={"id": "player_id"}),
        on="player_id",
        how="left",
    )

    top = projections.sort_values("vorp", ascending=False).head(30)
    print("\n── Top 30 players by VORP ──")
    for _, row in top.iterrows():
        print(
            f"  {names_map.get(int(row['player_id']), '?'):>18s}  "
            f"pos={int(row['element_type'])}  "
            f"xMin={row['xmin']:5.1f}  "
            f"xQual={row['xp_quality']:5.2f}  "
            f"xP={row['xp_final']:5.2f}  "
            f"VORP={row['vorp']:+.2f}"
        )

    return projections


async def cmd_draft(cfg: FPLConfig, pick_position: int) -> None:
    """Run Monte Carlo draft simulation."""
    log.info("▸ Running draft simulation from pick position %d", pick_position)

    projections = await cmd_pipeline(cfg)

    names_map: dict[int, str] = {}
    if "web_name" in projections.columns:
        names_map = dict(
            zip(projections["player_id"].astype(int), projections["web_name"])
        )

    pool = build_player_pool(projections, names_map)
    sim = DraftSimulator(pool, cfg)

    my_picks, final_state = sim.simulate(
        my_manager=pick_position - 1,
        n_sims=cfg.MC_SIMULATIONS,
    )

    pick_counts = cp.zeros(pool.n_players, dtype=cp.int32)
    for col in range(my_picks.shape[1]):
        indices, counts = cp.unique(my_picks[:, col], return_counts=True)
        pick_counts[indices] += counts

    top_idx = cp.argsort(-pick_counts)[:20].get()
    print(f"\n── Most-drafted players across {cfg.MC_SIMULATIONS:,} simulations (pick #{pick_position}) ──")
    for idx in top_idx:
        pct = float(pick_counts[idx]) / cfg.MC_SIMULATIONS * 100
        print(
            f"  {pool.names[idx]:>18s}  "
            f"VORP={float(pool.vorp[idx]):+.2f}  "
            f"drafted {pct:5.1f}%"
        )

    watchlist = list(cp.argsort(-pool.vorp)[:10].get())
    recs = sim.recommend(
        my_manager=pick_position - 1,
        current_pick=0,
        watchlist=watchlist,
    )
    print("\n── Survival probabilities for top-VORP watchlist ──")
    for r in recs:
        flag = " ⚑ PRIORITY" if r["priority"] else ""
        print(
            f"  {r['name']:>18s}  "
            f"VORP={r['vorp']:+.2f}  "
            f"P(survive)={r['survival_prob']:.1%}{flag}"
        )


async def cmd_waiver(
    cfg: FPLConfig,
    current_gw: int,
    roster_ids: list[int],
) -> None:
    """Weekly waiver analysis."""
    log.info("▸ Running waiver analysis for GW %d", current_gw)

    players, history, fixtures = await run_ingestion(cfg)
    projections = run_predictions(history, cfg)

    projections = projections.merge(
        players[["id", "team", "web_name"]].rename(columns={"id": "player_id"}),
        on="player_id",
        how="left",
    )

    names_map: dict[int, str] = dict(
        zip(
            players["id"].tolist(),
            players["web_name"].tolist(),
        )
    )

    roster = pd.DataFrame({"player_id": roster_ids})

    all_rostered = roster_ids
    engine = WaiverEngine(cfg)
    moves = engine.recommend(
        roster, projections, all_rostered, fixtures, current_gw, names_map,
    )

    print(f"\n── Waiver recommendations (GW {current_gw}, horizon={cfg.WAIVER_HORIZON_GW}) ──")
    if not moves:
        print("  No beneficial moves found.")
    for m in moves:
        print(
            f"  DROP {m.drop_name:>18s} (xMin={m.drop_xmin:.0f})  →  "
            f"ADD {m.add_name:>18s}  "
            f"net VORP = {m.net_vorp_gain:+.2f}"
        )

    fa = evaluate_free_agents(projections, all_rostered, fixtures, current_gw, cfg)
    streamers = find_streaming_defenders(fa)
    if len(streamers) > 0:
        print("\n── Top streaming defenders / GKPs ──")
        for _, row in streamers.iterrows():
            print(
                f"  {names_map.get(int(row['player_id']), '?'):>18s}  "
                f"pos={int(row['element_type'])}  "
                f"fixture_attract={row['fixture_attract']:.2f}  "
                f"stream_score={row['stream_score']:.3f}"
            )


# ═══════════════════════════════════════════════════════════════
# Argument parser
# ═══════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FPL Draft Optimisation Engine — GPU-accelerated",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python main.py pipeline                          Full ingest -> VORP rankings\n"
            "  python main.py draft --pick-position 3           Monte Carlo from seat #3\n"
            "  python main.py waiver --current-gw 12 --roster-ids 10,44,87,102\n"
        ),
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("pipeline", help="Full ingest -> project -> VORP pipeline")

    draft_p = sub.add_parser("draft", help="Monte Carlo draft simulation")
    draft_p.add_argument(
        "--pick-position", type=int, default=1,
        help="Your 1-indexed seat at the draft table (1-16).",
    )

    waiver_p = sub.add_parser("waiver", help="Weekly waiver analysis")
    waiver_p.add_argument("--current-gw", type=int, required=True)
    waiver_p.add_argument(
        "--roster-ids", type=str, required=True,
        help="Comma-separated FPL player IDs on your roster.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "pipeline":
        asyncio.run(cmd_pipeline(CONF))

    elif args.command == "draft":
        asyncio.run(cmd_draft(CONF, args.pick_position))

    elif args.command == "waiver":
        ids = [int(x.strip()) for x in args.roster_ids.split(",")]
        asyncio.run(cmd_waiver(CONF, args.current_gw, ids))


if __name__ == "__main__":
    main()
