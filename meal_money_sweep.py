#!/usr/bin/env python3
"""Parameter search for Meal Money production candidates."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from dataclasses import replace
from datetime import datetime

import pandas as pd

from strategy.meal_money import (
    BUY_COST_RATE,
    STOCK_TRANSACTION_TAX_RATE,
    MealMoneyConfig,
    load_intraday_data,
    prepare_backtest_context,
    run_backtest_from_context,
)


ENTRY_WINDOWS = {
    "morning_strict": ("09:15", "09:35", "09:40"),
    "ten": ("10:00", "10:20", "10:25"),
    "eleven": ("11:00", "11:20", "11:25"),
    "eleven_forty": ("11:40", "12:00", "12:05"),
    "noon": ("12:00", "12:20", "12:25"),
    "twelve_fifty": ("12:50", "13:10", "13:15"),
    "one": ("13:00", "13:20", "13:25"),
}


SECTOR_SETS = {
    "all": (),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep Meal Money v2 configs")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--pool", choices=["default", "extended", "all"], default="extended")
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--start-date", default="2024-05-01")
    parser.add_argument("--split-date", default="2025-05-01")
    parser.add_argument("--end-date", default="2026-04-01")
    parser.add_argument("--max-configs", type=int, default=240)
    parser.add_argument("--min-test-trades", type=int, default=25)
    parser.add_argument("--min-test-success-rate", type=float, default=0.45)
    parser.add_argument("--min-test-avg-pnl", type=float, default=0.0)
    parser.add_argument("--require-train-profit", action="store_true", default=True)
    parser.add_argument("--allow-no-train-profit", dest="require_train_profit", action="store_false")
    return parser.parse_args()


def resolve_tickers(args: argparse.Namespace) -> list[str] | None:
    if args.tickers:
        return args.tickers
    if args.pool == "all":
        return None
    try:
        from ai_report import DEFAULT_TICKERS, EXTENDED_TICKERS
        return DEFAULT_TICKERS if args.pool == "default" else EXTENDED_TICKERS
    except Exception:
        return None


def cost_pair(discount: float) -> tuple[float, float]:
    buy = BUY_COST_RATE * discount
    sell = BUY_COST_RATE * discount + STOCK_TRANSACTION_TAX_RATE
    return buy, sell


def config_grid(max_configs: int) -> list[tuple[str, str, float, MealMoneyConfig]]:
    """Build a bounded deterministic config grid."""
    windows = list(ENTRY_WINDOWS.items())
    price_bands = [(103, 150), (103, 180), (120, 180), (130, 180)]
    thresholds = [2.0, 2.5, 3.0, 3.5]
    gap_ranges = [(-0.010, 0.020), (-0.005, 0.025), (0.000, 0.030), (0.002, 0.035)]
    min_volumes = [2_000_000, 5_000_000, 8_000_000, 12_000_000]
    entry_volumes = [100_000, 200_000, 300_000, 500_000]
    pre_entry_volumes = [0, 500_000, 1_000_000, 2_000_000]
    pre_entry_moves = [0.0, 0.005, 0.010, 0.015]
    pre_entry_ranges = [0.0, 0.010, 0.015, 0.020]
    market_filters = [
        (False, 0.00, -1.000),
        (True, 0.25, -0.050),
        (True, 0.35, -0.030),
        (True, 0.40, -0.020),
    ]
    liquidity_weights = [0.0, 0.5, 1.2, 1.8]
    follow_weights = [0.0, 0.2, 0.7]
    stop_losses = [0.015, 0.020, 0.030]
    target_mins = [500, 600]
    required_ticks = [8, 10, 12, 16]
    edge_ticks = [0, 1]
    sector_sets = list(SECTOR_SETS.items())
    discounts = [1.0]

    raw = itertools.product(
        price_bands, thresholds, gap_ranges, min_volumes,
        entry_volumes, pre_entry_volumes, pre_entry_moves, pre_entry_ranges,
        stop_losses, target_mins, required_ticks, edge_ticks,
        sector_sets, discounts, market_filters, liquidity_weights, follow_weights,
    )

    prod_buy, prod_sell = cost_pair(1.0)
    configs = [(
        "morning_strict",
        "all",
        1.0,
        MealMoneyConfig(
            target_min=500,
            target_max=800,
            min_trade_capital=15_000,
            trade_capital=100_000,
            max_trade_capital=100_000,
            threshold=2.0,
            min_price=103,
            max_price=180,
            min_avg_volume=2_000_000,
            min_entry_bar_volume=0,
            min_pre_entry_volume=2_000_000,
            min_pre_entry_move_pct=0.020,
            min_pre_entry_range_pct=0.015,
            use_market_filter=False,
            score_rank_weight=1.0,
            liquidity_rank_weight=0.5,
            market_follow_rank_weight=0.2,
            entry_time="09:15",
            force_exit_time="09:35",
            market_close_cutoff="09:40",
            min_open_gap_pct=0.005,
            max_open_gap_pct=0.040,
            max_required_move_pct=0.045,
            max_required_ticks=3,
            entry_edge_ticks=1,
            stop_loss_pct=0.005,
            buy_cost=prod_buy,
            sell_cost=prod_sell,
            transaction_tax=STOCK_TRANSACTION_TAX_RATE,
            min_commission=20.0,
            slippage=0.0,
            lot_size=1,
            allowed_sectors=(),
        ),
    )]
    if len(configs) >= max_configs:
        return configs
    for idx, (
        (min_price, max_price),
        threshold,
        (min_gap, max_gap),
        min_avg_volume,
        min_entry_bar_volume,
        min_pre_entry_volume,
        min_pre_entry_move_pct,
        min_pre_entry_range_pct,
        stop_loss,
        target_min,
        max_required_ticks,
        entry_edge_ticks,
        (sector_name, allowed_sectors),
        discount,
        (use_market_filter, market_breadth, market_return),
        liquidity_weight,
        follow_weight,
    ) in enumerate(raw):
        # Deterministic thinning keeps coverage across dimensions without running
        # every Cartesian product.
        if idx % 37 not in {0, 3, 11, 19, 29}:
            continue
        for window_name, (entry_time, force_exit_time, cutoff) in windows:
            buy_cost, sell_cost = cost_pair(discount)
            config = MealMoneyConfig(
                target_min=target_min,
                target_max=max(target_min, 800),
                min_trade_capital=15_000,
                trade_capital=100_000,
                max_trade_capital=100_000,
                threshold=threshold,
                min_price=min_price,
                max_price=max_price,
                min_avg_volume=min_avg_volume,
                min_entry_bar_volume=min_entry_bar_volume,
                min_pre_entry_volume=min_pre_entry_volume,
                min_pre_entry_move_pct=min_pre_entry_move_pct,
                min_pre_entry_range_pct=min_pre_entry_range_pct,
                use_market_filter=use_market_filter,
                min_market_breadth_20=market_breadth,
                min_market_return_5d=market_return,
                score_rank_weight=1.0,
                liquidity_rank_weight=liquidity_weight,
                market_follow_rank_weight=follow_weight,
                entry_time=entry_time,
                force_exit_time=force_exit_time,
                market_close_cutoff=cutoff,
                min_open_gap_pct=min_gap,
                max_open_gap_pct=max_gap,
                max_required_move_pct=0.060,
                max_required_ticks=max_required_ticks,
                entry_edge_ticks=entry_edge_ticks,
                stop_loss_pct=stop_loss,
                buy_cost=buy_cost,
                sell_cost=sell_cost,
                transaction_tax=STOCK_TRANSACTION_TAX_RATE,
                min_commission=20.0,
                slippage=0.0,
                lot_size=1,
                allowed_sectors=allowed_sectors,
            )
            configs.append((window_name, sector_name, discount, config))
            if len(configs) >= max_configs:
                return configs
    return configs


def metrics(prefix: str, summary: dict) -> dict:
    return {
        f"{prefix}_active_days": summary["active_days"],
        f"{prefix}_success_days": summary["success_days"],
        f"{prefix}_success_rate": summary["daily_success_rate"],
        f"{prefix}_trades": summary["total_trades"],
        f"{prefix}_target_rate": summary["trade_target_rate"],
        f"{prefix}_win_rate": summary["win_rate"],
        f"{prefix}_total_pnl": summary["total_pnl"],
        f"{prefix}_avg_trade_pnl": summary["avg_trade_pnl"],
        f"{prefix}_cutoff_violations": summary["cutoff_violations"],
    }


def passes_gate(row: dict, args: argparse.Namespace) -> bool:
    if row["test_trades"] < args.min_test_trades:
        return False
    if row["test_total_pnl"] <= 0:
        return False
    if row["test_avg_trade_pnl"] < args.min_test_avg_pnl:
        return False
    if row["test_success_rate"] < args.min_test_success_rate:
        return False
    if row["test_cutoff_violations"] != 0:
        return False
    if args.require_train_profit and row["train_total_pnl"] <= 0:
        return False
    return True


def main() -> int:
    args = parse_args()
    tickers = resolve_tickers(args)
    print(f"Loading intraday data: pool={args.pool}, tickers={len(tickers) if tickers else 'all'}")
    intraday = load_intraday_data(args.data_dir, tickers)
    if not intraday:
        print("No intraday data loaded.")
        return 1

    base_config = MealMoneyConfig()
    context = prepare_backtest_context(intraday, base_config)

    rows = []
    configs = config_grid(args.max_configs)
    print(f"Running {len(configs)} configs...")
    train_end = (pd.Timestamp(args.split_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    for idx, (window_name, sector_name, discount, config) in enumerate(configs, 1):
        _, _, train_summary = run_backtest_from_context(
            intraday, context, config,
            start_date=args.start_date, end_date=train_end,
        )
        _, _, test_summary = run_backtest_from_context(
            intraday, context, config,
            start_date=args.split_date, end_date=args.end_date,
        )
        _, _, full_summary = run_backtest_from_context(
            intraday, context, config,
            start_date=args.start_date, end_date=args.end_date,
        )
        row = {
            "config_id": idx,
            "window": window_name,
            "sector_set": sector_name,
            "commission_discount": discount,
            "entry_time": config.entry_time,
            "force_exit_time": config.force_exit_time,
            "target_min": config.target_min,
            "threshold": config.threshold,
            "min_price": config.min_price,
            "max_price": config.max_price,
            "min_gap_pct": config.min_open_gap_pct,
            "max_gap_pct": config.max_open_gap_pct,
            "min_avg_volume": config.min_avg_volume,
            "min_entry_bar_volume": config.min_entry_bar_volume,
            "min_pre_entry_volume": config.min_pre_entry_volume,
            "min_pre_entry_move_pct": config.min_pre_entry_move_pct,
            "min_pre_entry_range_pct": config.min_pre_entry_range_pct,
            "use_market_filter": config.use_market_filter,
            "min_market_breadth_20": config.min_market_breadth_20,
            "min_market_return_5d": config.min_market_return_5d,
            "liquidity_rank_weight": config.liquidity_rank_weight,
            "market_follow_rank_weight": config.market_follow_rank_weight,
            "stop_loss_pct": config.stop_loss_pct,
            "max_required_ticks": config.max_required_ticks,
            "entry_edge_ticks": config.entry_edge_ticks,
            "buy_cost": config.buy_cost,
            "sell_cost": config.sell_cost,
            "transaction_tax": config.transaction_tax,
            "min_commission": config.min_commission,
            "slippage": config.slippage,
            **metrics("train", train_summary),
            **metrics("test", test_summary),
            **metrics("full", full_summary),
            "config_json": json.dumps(config.to_dict(), ensure_ascii=False),
        }
        row["production_gate"] = passes_gate(row, args)
        row["score"] = (
            row["test_total_pnl"]
            + 0.5 * row["train_total_pnl"]
            + 5000 * row["test_success_rate"]
            - 1000 * max(0, 1.0 - row["test_trades"] / max(args.min_test_trades, 1))
        )
        rows.append(row)
        if idx % 25 == 0:
            best = max(rows, key=lambda x: x["score"])
            print(
                f"  {idx}/{len(configs)} best={best['config_id']} "
                f"test_pnl={best['test_total_pnl']:+.0f} "
                f"test_sr={best['test_success_rate']:.2%} gate={best['production_gate']}"
            )

    results = pd.DataFrame(rows).sort_values(
        ["production_gate", "score"], ascending=[False, False]
    )
    os.makedirs("artifacts", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"artifacts/meal_money_sweep_{stamp}.csv"
    out_json = f"artifacts/meal_money_sweep_best_{stamp}.json"
    results.to_csv(out_csv, index=False)
    best = results.iloc[0].to_dict() if not results.empty else {}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False, default=str)

    print("\nTop 10:")
    cols = [
        "config_id", "production_gate", "window", "sector_set",
        "commission_discount", "entry_time", "target_min", "threshold",
        "min_avg_volume", "min_entry_bar_volume", "min_pre_entry_volume",
        "min_pre_entry_move_pct", "min_pre_entry_range_pct",
        "use_market_filter", "min_market_breadth_20",
        "test_trades", "test_total_pnl", "test_success_rate",
        "test_avg_trade_pnl", "train_total_pnl", "full_total_pnl",
    ]
    print(results[cols].head(10).to_string(index=False))
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
