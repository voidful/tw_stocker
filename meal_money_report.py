#!/usr/bin/env python3
"""CLI for the meal-money morning strategy."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

from strategy.meal_money import (
    MealMoneyConfig,
    build_latest_watchlist,
    compute_time_focus_stats,
    load_intraday_data,
    load_night_market_csv,
    run_backtest,
    summarize_by_sector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Meal Money v1 - morning strategy ending before 09:40"
    )
    parser.add_argument(
        "--mode", choices=["backtest", "signals"], default="backtest",
        help="backtest local 5-minute data or generate next-session watchlist",
    )
    parser.add_argument("--data-dir", default="data", help="local 5-minute CSV directory")
    parser.add_argument("--pool", choices=["default", "extended", "all"], default="extended")
    parser.add_argument("--tickers", nargs="+", default=None, help="optional ticker subset")
    parser.add_argument("--start-date", default=None, help="backtest start date YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="backtest end date YYYY-MM-DD")
    parser.add_argument("--target-min", type=float, default=500.0)
    parser.add_argument("--target-max", type=float, default=800.0)
    parser.add_argument("--min-trade-capital", type=float, default=15_000.0)
    parser.add_argument("--trade-capital", type=float, default=100_000.0)
    parser.add_argument("--max-trade-capital", type=float, default=100_000.0)
    parser.add_argument("--max-trades-per-day", type=int, default=1)
    parser.add_argument("--universe-size", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--min-price", type=float, default=103.0)
    parser.add_argument("--max-price", type=float, default=180.0)
    parser.add_argument("--min-avg-volume", type=float, default=2_000_000.0)
    parser.add_argument("--min-entry-bar-volume", type=float, default=0.0)
    parser.add_argument("--pre-entry-start-time", default="09:00")
    parser.add_argument("--pre-entry-end-time", default="09:10")
    parser.add_argument("--min-pre-entry-volume", type=float, default=2_000_000.0)
    parser.add_argument("--min-pre-entry-move-pct", type=float, default=0.020)
    parser.add_argument("--min-pre-entry-range-pct", type=float, default=0.015)
    parser.add_argument("--market-filter", action="store_true", default=False,
                        help="require local market breadth/return context to be tradable")
    parser.add_argument("--no-market-filter", action="store_false", dest="market_filter")
    parser.add_argument("--min-market-breadth-20", type=float, default=0.40)
    parser.add_argument("--min-market-return-5d", type=float, default=-0.020)
    parser.add_argument("--score-rank-weight", type=float, default=1.0)
    parser.add_argument("--liquidity-rank-weight", type=float, default=0.5)
    parser.add_argument("--market-follow-rank-weight", type=float, default=0.2)
    parser.add_argument("--entry-time", default="09:15")
    parser.add_argument("--force-exit-time", default="09:35")
    parser.add_argument("--min-open-gap-pct", type=float, default=0.005,
                        help="minimum opening gap, decimal form")
    parser.add_argument("--max-open-gap-pct", type=float, default=0.040,
                        help="maximum opening gap, decimal form")
    parser.add_argument("--max-required-move-pct", type=float, default=0.045,
                        help="skip trades whose fee-adjusted target needs a larger move")
    parser.add_argument("--max-required-ticks", type=int, default=3)
    parser.add_argument("--entry-edge-ticks", type=int, default=1)
    parser.add_argument("--stop-loss-pct", type=float, default=0.005)
    parser.add_argument("--buy-cost", type=float, default=0.001425,
                        help="buy commission rate")
    parser.add_argument("--sell-cost", type=float, default=0.004425,
                        help="sell commission rate + stock transaction tax rate")
    parser.add_argument("--transaction-tax", type=float, default=0.003,
                        help="stock transaction tax rate charged on sell notional")
    parser.add_argument("--min-commission", type=float, default=20.0,
                        help="minimum commission per side")
    parser.add_argument("--slippage", type=float, default=0.0)
    parser.add_argument("--lot-size", type=int, default=1)
    parser.add_argument(
        "--allowed-sectors", nargs="*",
        default=[],
        help="optional sector keys allowed for entry; default is no sector restriction",
    )
    parser.add_argument("--track-us-market", action="store_true",
                        help="fetch SOX/SPY/VIX context and write it to output columns")
    parser.add_argument("--use-us-market", action="store_true",
                        help="fetch and gate by latest available SOX/SPY/VIX context")
    parser.add_argument("--min-sox-1d", type=float, default=-0.02)
    parser.add_argument("--min-sox-5d", type=float, default=-0.05)
    parser.add_argument("--night-market-csv", default=None,
                        help="optional CSV with Date,Night_Return_Pct")
    parser.add_argument("--use-night-filter", action="store_true",
                        help="require night-market CSV context to pass")
    parser.add_argument("--min-night-return-pct", type=float, default=-0.4)
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


def make_config(args: argparse.Namespace) -> MealMoneyConfig:
    return MealMoneyConfig(
        target_min=args.target_min,
        target_max=args.target_max,
        min_trade_capital=args.min_trade_capital,
        trade_capital=args.trade_capital,
        max_trade_capital=args.max_trade_capital,
        max_trades_per_day=args.max_trades_per_day,
        top_n_universe=args.universe_size,
        top_k_candidates=args.top_k,
        threshold=args.threshold,
        min_price=args.min_price,
        max_price=args.max_price,
        min_avg_volume=args.min_avg_volume,
        min_entry_bar_volume=args.min_entry_bar_volume,
        pre_entry_start_time=args.pre_entry_start_time,
        pre_entry_end_time=args.pre_entry_end_time,
        min_pre_entry_volume=args.min_pre_entry_volume,
        min_pre_entry_move_pct=args.min_pre_entry_move_pct,
        min_pre_entry_range_pct=args.min_pre_entry_range_pct,
        use_market_filter=args.market_filter,
        min_market_breadth_20=args.min_market_breadth_20,
        min_market_return_5d=args.min_market_return_5d,
        score_rank_weight=args.score_rank_weight,
        liquidity_rank_weight=args.liquidity_rank_weight,
        market_follow_rank_weight=args.market_follow_rank_weight,
        entry_time=args.entry_time,
        force_exit_time=args.force_exit_time,
        min_open_gap_pct=args.min_open_gap_pct,
        max_open_gap_pct=args.max_open_gap_pct,
        max_required_move_pct=args.max_required_move_pct,
        max_required_ticks=args.max_required_ticks,
        entry_edge_ticks=args.entry_edge_ticks,
        stop_loss_pct=args.stop_loss_pct,
        buy_cost=args.buy_cost,
        sell_cost=args.sell_cost,
        transaction_tax=args.transaction_tax,
        min_commission=args.min_commission,
        slippage=args.slippage,
        lot_size=args.lot_size,
        use_sox_filter=args.use_us_market,
        min_sox_1d=args.min_sox_1d,
        min_sox_5d=args.min_sox_5d,
        allowed_sectors=tuple(args.allowed_sectors),
        use_night_filter=args.use_night_filter,
        min_night_return_pct=args.min_night_return_pct,
    )


def print_summary(summary: dict) -> None:
    print("Meal Money v1 summary")
    print(f"  Active days:              {summary['active_days']}")
    print(f"  Success days:             {summary['success_days']}")
    print(f"  Daily success rate:       {summary['daily_success_rate'] * 100:.1f}%")
    print(f"  Total trades:             {summary['total_trades']}")
    print(f"  Target-hit trades:        {summary['target_trades']}")
    print(f"  Trade target rate:        {summary['trade_target_rate'] * 100:.1f}%")
    print(f"  Win rate:                 {summary['win_rate'] * 100:.1f}%")
    print(f"  Avg trades / active day:  {summary['avg_trades_per_active_day']:.2f}")
    print(f"  Avg trades / success day: {summary['avg_trades_per_success_day']:.2f}")
    print(f"  Total net PnL:            {summary['total_pnl']:+,.0f}")
    print(f"  Avg trade net PnL:        {summary['avg_trade_pnl']:+,.0f}")
    print(f"  Exit >= 09:40 violations: {summary['cutoff_violations']}")


def main() -> int:
    args = parse_args()
    config = make_config(args)

    tickers = resolve_tickers(args)
    intraday = load_intraday_data(args.data_dir, tickers)
    if not intraday:
        print("No intraday data loaded.")
        return 1

    os.makedirs("artifacts", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    night_market = load_night_market_csv(args.night_market_csv) if args.night_market_csv else None
    us_signals = None
    if args.track_us_market or args.use_us_market:
        try:
            from strategy.us_market import fetch_us_signals
            us_signals = fetch_us_signals(
                start_date=args.start_date, end_date=args.end_date, days=1500
            )
        except Exception as exc:
            print(f"US market context unavailable: {exc}")
            if config.use_sox_filter:
                print("SOX gate requested, so no trades will be eligible without US context.")

    if args.mode == "signals":
        watchlist = build_latest_watchlist(intraday, config)
        out_csv = f"artifacts/meal_money_watchlist_{stamp}.csv"
        out_json = f"artifacts/meal_money_orders_{stamp}.json"
        watchlist.to_csv(out_csv, index=False)
        payload = {
            "created_at": datetime.now().isoformat(),
            "strategy_version": "meal_money_v1",
            "config": config.to_dict(),
            "orders": watchlist.to_dict(orient="records"),
            "execution_rule": (
                "Enter at 09:05 open only if the actual opening price remains "
                "inside Valid_Entry_Min/Max; place fee-adjusted net target order; "
                "force exit at 09:35."
            ),
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        if watchlist.empty:
            print("No meal-money watchlist candidates.")
        else:
            print(watchlist.to_string(index=False))
        print(f"Saved: {out_csv}")
        print(f"Saved: {out_json}")
        return 0

    trades_df, daily_df, summary = run_backtest(
        intraday, config, start_date=args.start_date, end_date=args.end_date,
        us_signals=us_signals, night_market=night_market,
    )
    out_trades = f"artifacts/meal_money_trades_{stamp}.csv"
    out_daily = f"artifacts/meal_money_daily_{stamp}.csv"
    out_sector = f"artifacts/meal_money_sector_{stamp}.csv"
    out_time = f"artifacts/meal_money_time_focus_{stamp}.csv"
    out_meta = f"artifacts/meal_money_metadata_{stamp}.json"
    sector_df = summarize_by_sector(trades_df)
    time_df = compute_time_focus_stats(intraday, config)
    trades_df.to_csv(out_trades, index=False)
    daily_df.to_csv(out_daily, index=False)
    sector_df.to_csv(out_sector, index=False)
    time_df.to_csv(out_time, index=False)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump({
            "created_at": datetime.now().isoformat(),
            "strategy_version": "meal_money_v1",
            "summary": summary,
        }, f, indent=2, ensure_ascii=False)

    print_summary(summary)
    print(f"Saved: {out_trades}")
    print(f"Saved: {out_daily}")
    print(f"Saved: {out_sector}")
    print(f"Saved: {out_time}")
    print(f"Saved: {out_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
