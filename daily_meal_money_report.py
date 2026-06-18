#!/usr/bin/env python3
"""CLI for the daily bidirectional meal-money strategy."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import pandas as pd

from strategy.daily_meal_money import (
    DEFAULT_MEAL_TRADING_WINDOWS,
    DailyMealMoneyConfig,
    build_daily_features,
    load_intraday_data,
    run_daily_backtest,
    select_daily_candidate_pool,
    select_daily_candidates,
    simulate_daily_candidate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Daily Meal Money - bidirectional high-volume day strategy"
    )
    parser.add_argument(
        "--preset",
        choices=["daily", "small-cap-daily", "small-cap-one-shot"],
        default="small-cap-daily",
        help=(
            "strategy preset; small-cap-daily is the default 20k cap strategy, "
            "daily keeps the historical 100k reference, small-cap-one-shot "
            "targets fewer 20k trades"
        ),
    )
    parser.add_argument("--mode", choices=["backtest", "signals"], default="backtest")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--pool", choices=["default", "extended", "all"], default="all")
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--target-min", type=float, default=600.0)
    parser.add_argument("--trade-capital", type=float, default=100_000.0)
    parser.add_argument("--max-trade-capital", type=float, default=100_000.0)
    parser.add_argument("--min-trade-capital", type=float, default=15_000.0)
    parser.add_argument("--min-price", type=float, default=103.0)
    parser.add_argument("--max-price", type=float, default=180.0)
    parser.add_argument("--min-avg-volume", type=float, default=500_000.0)
    parser.add_argument("--min-pre-entry-volume", type=float, default=1_000_000.0)
    parser.add_argument("--min-abs-pre-entry-move-pct", type=float, default=0.0)
    parser.add_argument("--min-pre-entry-range-pct", type=float, default=0.015)
    parser.add_argument("--rel-pre-entry-volume-cap", type=float, default=99.0)
    parser.add_argument("--long-gap-min", type=float, default=-0.015)
    parser.add_argument("--long-gap-max", type=float, default=0.025)
    parser.add_argument("--short-gap-min", type=float, default=-0.025)
    parser.add_argument("--short-gap-max", type=float, default=0.015)
    parser.add_argument("--max-required-ticks", type=int, default=4)
    parser.add_argument("--entry-edge-ticks", type=int, default=1)
    parser.add_argument("--stop-loss-pct", type=float, default=0.020)
    parser.add_argument("--transaction-tax", type=float, default=0.0015)
    parser.add_argument("--buy-cost", type=float, default=0.001425)
    parser.add_argument("--sell-commission", type=float, default=0.001425)
    parser.add_argument("--min-commission", type=float, default=20.0)
    parser.add_argument("--slippage", type=float, default=0.0)
    parser.add_argument("--lot-size", type=int, default=1)
    parser.add_argument(
        "--score-mode",
        choices=[
            "volume_first", "pressure", "move_first", "range_first",
            "abs_pressure", "follow_prev",
        ],
        default="abs_pressure",
    )
    parser.add_argument("--short-score-bias", type=float, default=30.0)
    parser.add_argument("--reselect-after-feasibility", action="store_true")
    parser.add_argument("--use-meal-windows", action="store_true")
    parser.add_argument("--non-morning-window-score-penalty", type=float, default=0.0)
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--short-only", action="store_true")
    return parser.parse_args()


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.preset == "small-cap-daily":
        args.target_min = 420.0
        args.min_trade_capital = 15_000.0
        args.trade_capital = 20_000.0
        args.max_trade_capital = 20_000.0
        args.min_avg_volume = 1_000_000.0
        args.min_pre_entry_volume = 500_000.0
        args.min_abs_pre_entry_move_pct = 0.0
        args.min_pre_entry_range_pct = 0.012
        args.rel_pre_entry_volume_cap = 2.0
        args.max_required_ticks = 6
        args.stop_loss_pct = 0.020
        args.transaction_tax = 0.0015
        args.score_mode = "abs_pressure"
        args.short_score_bias = 10.0
        args.reselect_after_feasibility = True
        args.use_meal_windows = True
        args.non_morning_window_score_penalty = 20.0
        args.long_only = False
        args.short_only = False
        return args
    if args.preset != "small-cap-one-shot":
        return args
    args.target_min = 500.0
    args.min_trade_capital = 15_000.0
    args.trade_capital = 20_000.0
    args.max_trade_capital = 20_000.0
    args.min_avg_volume = 500_000.0
    args.min_pre_entry_volume = 500_000.0
    args.min_abs_pre_entry_move_pct = 0.030
    args.min_pre_entry_range_pct = 0.060
    args.rel_pre_entry_volume_cap = 2.0
    args.max_required_ticks = 12
    args.stop_loss_pct = 0.020
    args.transaction_tax = 0.0015
    args.score_mode = "move_first"
    args.short_score_bias = 30.0
    args.reselect_after_feasibility = False
    args.use_meal_windows = False
    args.non_morning_window_score_penalty = 0.0
    args.long_only = False
    args.short_only = True
    return args


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


def make_config(args: argparse.Namespace) -> DailyMealMoneyConfig:
    allow_long = not args.short_only
    allow_short = not args.long_only
    return DailyMealMoneyConfig(
        target_min=args.target_min,
        min_trade_capital=args.min_trade_capital,
        trade_capital=args.trade_capital,
        max_trade_capital=args.max_trade_capital,
        min_price=args.min_price,
        max_price=args.max_price,
        min_avg_volume=args.min_avg_volume,
        min_pre_entry_volume=args.min_pre_entry_volume,
        min_abs_pre_entry_move_pct=args.min_abs_pre_entry_move_pct,
        min_pre_entry_range_pct=args.min_pre_entry_range_pct,
        rel_pre_entry_volume_cap=args.rel_pre_entry_volume_cap,
        long_gap_min=args.long_gap_min,
        long_gap_max=args.long_gap_max,
        short_gap_min=args.short_gap_min,
        short_gap_max=args.short_gap_max,
        max_required_ticks=args.max_required_ticks,
        entry_edge_ticks=args.entry_edge_ticks,
        stop_loss_pct=args.stop_loss_pct,
        buy_cost=args.buy_cost,
        sell_cost=args.sell_commission + args.transaction_tax,
        transaction_tax=args.transaction_tax,
        min_commission=args.min_commission,
        slippage=args.slippage,
        lot_size=args.lot_size,
        allow_long=allow_long,
        allow_short=allow_short,
        score_mode=args.score_mode,
        short_score_bias=args.short_score_bias,
        reselect_after_feasibility=args.reselect_after_feasibility,
        trading_windows=DEFAULT_MEAL_TRADING_WINDOWS if args.use_meal_windows else (),
        non_morning_window_score_penalty=args.non_morning_window_score_penalty,
    )


def print_summary(summary: dict) -> None:
    print("Daily Meal Money summary")
    print(f"  Active days:              {summary['active_days']}")
    print(f"  Active ratio:             {summary['active_ratio'] * 100:.1f}%")
    print(f"  Total trades:             {summary['total_trades']}")
    print(f"  Target-hit trades:        {summary['target_trades']}")
    print(f"  Target rate:              {summary['target_rate'] * 100:.1f}%")
    print(f"  Win rate:                 {summary['win_rate'] * 100:.1f}%")
    print(f"  Long ratio:               {summary['long_ratio'] * 100:.1f}%")
    print(f"  Total net PnL:            {summary['total_pnl']:+,.0f}")
    print(f"  Avg trade net PnL:        {summary['avg_trade_pnl']:+,.0f}")
    print(f"  Exit >= 09:40 violations: {summary['cutoff_violations']}")


def main() -> int:
    args = apply_preset(parse_args())
    config = make_config(args)
    tickers = resolve_tickers(args)
    print(
        "Loading intraday data: "
        f"pool={args.pool}, tickers={len(tickers) if tickers else 'all'}"
    )
    intraday = load_intraday_data(args.data_dir, tickers)
    if not intraday:
        print("No intraday data loaded.")
        return 1

    os.makedirs("artifacts", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    if args.mode == "signals":
        features = build_daily_features(
            intraday, config, start_date=args.start_date, end_date=args.end_date
        )
        trades = []
        if config.reselect_after_feasibility:
            candidates = select_daily_candidate_pool(features, config)
            for _, group in candidates.groupby("Date", sort=True):
                for _, row in group.iterrows():
                    trade = simulate_daily_candidate(row, config)
                    if trade is not None:
                        trades.append(trade)
                        break
            trades = trades[-10:]
        else:
            candidates = select_daily_candidates(features, config)
            for _, row in candidates.tail(10).iterrows():
                trade = simulate_daily_candidate(row, config)
                if trade is not None:
                    trades.append(trade)
        signals = pd.DataFrame(trades)
        out_csv = f"artifacts/daily_meal_money_signals_{stamp}.csv"
        signals.to_csv(out_csv, index=False)
        if signals.empty:
            print("No daily meal-money signal candidates.")
        else:
            cols = [
                "Date", "Ticker", "Window", "Side", "Entry_Price", "Target_Price",
                "Required_Ticks", "Pre_Volume", "Pre_Move_Pct",
                "Pre_Range_Pct", "Open_Gap_Pct", "Score",
            ]
            print(signals[cols].tail(10).to_string(index=False))
        print(f"Saved signals: {out_csv}")
        return 0

    trades, daily, summary = run_daily_backtest(
        intraday, config, start_date=args.start_date, end_date=args.end_date
    )
    print_summary(summary)
    out_trades = f"artifacts/daily_meal_money_trades_{stamp}.csv"
    out_daily = f"artifacts/daily_meal_money_daily_{stamp}.csv"
    out_summary = f"artifacts/daily_meal_money_summary_{stamp}.json"
    trades.to_csv(out_trades, index=False)
    daily.to_csv(out_daily, index=False)
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved trades:  {out_trades}")
    print(f"Saved daily:   {out_daily}")
    print(f"Saved summary: {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
