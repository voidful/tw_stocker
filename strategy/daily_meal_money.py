"""Daily bidirectional meal-money strategy.

This strategy is separate from ``strategy.meal_money`` because its objective is
different: it tries to produce one liquid long or short day-trade almost every
session, then exits before 09:40.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from strategy.meal_money import (
    COMMISSION_RATE,
    MIN_COMMISSION,
    MealMoneyConfig,
    buy_commission,
    calculate_shares,
    intraday_window,
    load_intraday_data,
    net_pnl,
    read_intraday_csv,
    round_down_to_tick,
    round_up_to_tick,
    sell_commission,
    target_exit_price,
    tick_size,
    ticks_between,
    transaction_tax,
)


DAYTRADE_TRANSACTION_TAX = 0.0015
TradingWindow = tuple[str, str, str, str, str, str]

DEFAULT_MEAL_TRADING_WINDOWS: tuple[TradingWindow, ...] = (
    ("09:00-09:40", "09:00", "09:10", "09:15", "09:35", "09:40"),
    ("10:00-10:20", "10:00", "10:05", "10:10", "10:15", "10:20"),
    ("11:00-11:20", "11:00", "11:05", "11:10", "11:15", "11:20"),
    ("11:40-12:20", "11:40", "11:50", "11:55", "12:15", "12:20"),
    ("12:50-13:30", "12:50", "13:00", "13:05", "13:25", "13:30"),
)


@dataclass(frozen=True)
class DailyMealMoneyConfig:
    """Configuration for the daily bidirectional lunch-money strategy."""

    target_min: float = 600.0
    target_max: float = 800.0
    min_trade_capital: float = 15_000.0
    trade_capital: float = 100_000.0
    max_trade_capital: float = 100_000.0
    min_price: float = 103.0
    max_price: float = 180.0
    min_avg_volume: float = 500_000.0
    pre_entry_start_time: str = "09:00"
    pre_entry_end_time: str = "09:10"
    entry_time: str = "09:15"
    force_exit_time: str = "09:35"
    market_close_cutoff: str = "09:40"
    min_pre_entry_volume: float = 1_000_000.0
    min_abs_pre_entry_move_pct: float = 0.0
    min_pre_entry_range_pct: float = 0.015
    rel_pre_entry_volume_cap: float = 99.0
    long_gap_min: float = -0.015
    long_gap_max: float = 0.025
    short_gap_min: float = -0.025
    short_gap_max: float = 0.015
    max_required_ticks: int = 4
    entry_edge_ticks: int = 1
    stop_loss_pct: float = 0.020
    buy_cost: float = COMMISSION_RATE
    transaction_tax: float = DAYTRADE_TRANSACTION_TAX
    sell_cost: float = COMMISSION_RATE + DAYTRADE_TRANSACTION_TAX
    min_commission: float = MIN_COMMISSION
    slippage: float = 0.0
    lot_size: int = 1
    allow_long: bool = True
    allow_short: bool = True
    score_mode: str = "abs_pressure"
    short_score_bias: float = 30.0
    reselect_after_feasibility: bool = False
    trading_windows: tuple[TradingWindow, ...] = ()
    non_morning_window_score_penalty: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def as_meal_config(self) -> MealMoneyConfig:
        """Return a cost-compatible config for shared sizing helpers."""
        return MealMoneyConfig(
            target_min=self.target_min,
            target_max=self.target_max,
            min_trade_capital=self.min_trade_capital,
            trade_capital=self.trade_capital,
            max_trade_capital=self.max_trade_capital,
            entry_time=self.entry_time,
            force_exit_time=self.force_exit_time,
            market_close_cutoff=self.market_close_cutoff,
            entry_edge_ticks=self.entry_edge_ticks,
            stop_loss_pct=self.stop_loss_pct,
            buy_cost=self.buy_cost,
            sell_cost=self.sell_cost,
            transaction_tax=self.transaction_tax,
            min_commission=self.min_commission,
            slippage=self.slippage,
            lot_size=self.lot_size,
        )


def validate_daily_config(config: DailyMealMoneyConfig) -> None:
    if config.target_min <= 0:
        raise ValueError("target_min must be positive")
    if config.min_trade_capital < 14_000:
        raise ValueError("min_trade_capital must be at least 14000")
    if config.trade_capital < config.min_trade_capital:
        raise ValueError("trade_capital must be at least min_trade_capital")
    if config.trade_capital > config.max_trade_capital:
        raise ValueError("trade_capital must not exceed max_trade_capital")
    if config.min_price <= 0 or config.max_price <= config.min_price:
        raise ValueError("price band must be positive and ordered")
    if config.entry_time < "09:15":
        raise ValueError("entry_time must not be before 09:15")
    if config.force_exit_time >= config.market_close_cutoff:
        raise ValueError("force_exit_time must be before market_close_cutoff")
    if config.pre_entry_start_time >= config.entry_time:
        raise ValueError("pre_entry_start_time must be before entry_time")
    if config.pre_entry_end_time >= config.entry_time:
        raise ValueError("pre_entry_end_time must be before entry_time")
    if not config.allow_long and not config.allow_short:
        raise ValueError("at least one side must be enabled")
    if config.max_required_ticks < 1:
        raise ValueError("max_required_ticks must be at least 1")
    if not (0 < config.stop_loss_pct < 1):
        raise ValueError("stop_loss_pct must be between 0 and 1")
    if config.transaction_tax < 0 or config.buy_cost < 0 or config.sell_cost < 0:
        raise ValueError("cost rates must be non-negative")
    if config.non_morning_window_score_penalty < 0:
        raise ValueError("non_morning_window_score_penalty must be non-negative")
    if config.score_mode not in {
        "volume_first",
        "pressure",
        "move_first",
        "range_first",
        "abs_pressure",
        "follow_prev",
    }:
        raise ValueError("unsupported score_mode")
    for window in _trading_windows(config):
        if len(window) != 6:
            raise ValueError("trading windows must have 6 fields")
        _, pre_start, pre_end, entry_time, force_exit_time, cutoff_time = window
        if pre_start >= pre_end:
            raise ValueError("trading window pre-start must be before pre-end")
        if pre_end >= entry_time:
            raise ValueError("trading window pre-end must be before entry")
        if entry_time >= force_exit_time:
            raise ValueError("trading window entry must be before force exit")
        if force_exit_time >= cutoff_time:
            raise ValueError("trading window force exit must be before cutoff")


def _trading_windows(config: DailyMealMoneyConfig) -> tuple[TradingWindow, ...]:
    if config.trading_windows:
        return config.trading_windows
    return ((
        f"{config.pre_entry_start_time}-{config.market_close_cutoff}",
        config.pre_entry_start_time,
        config.pre_entry_end_time,
        config.entry_time,
        config.force_exit_time,
        config.market_close_cutoff,
    ),)


def short_net_pnl(entry_price: float, cover_price: float, shares: int,
                  config: DailyMealMoneyConfig) -> float:
    """Net PnL for same-day sell-first then buy-cover stock trade."""
    meal_config = config.as_meal_config()
    entry_gross = shares * entry_price
    cover_gross = shares * cover_price
    proceeds = (
        entry_gross
        - sell_commission(entry_price, shares, meal_config)
        - transaction_tax(entry_price, shares, meal_config)
        - entry_gross * config.slippage
    )
    cover_cost = (
        cover_gross
        + buy_commission(cover_price, shares, meal_config)
        + cover_gross * config.slippage
    )
    return proceeds - cover_cost


def target_cover_price(entry_price: float, shares: int,
                       config: DailyMealMoneyConfig) -> float:
    """Cover price required for a short trade to reach target_min net PnL."""
    if shares <= 0:
        return np.nan
    price = round_down_to_tick(entry_price)
    for _ in range(1000):
        if price <= 0:
            return np.nan
        if short_net_pnl(entry_price, price, shares, config) >= config.target_min:
            return price
        price = round_down_to_tick(price - tick_size(price))
    return np.nan


def short_ticks_between(entry_price: float, cover_price: float) -> int:
    step = tick_size(entry_price)
    return int(np.ceil(max(entry_price - cover_price, 0) / step - 1e-12))


def _bar_times(index: pd.Index) -> list[str]:
    return [pd.Timestamp(ts).strftime("%H:%M") for ts in index]


def build_daily_features(intraday: Dict[str, pd.DataFrame],
                         config: Optional[DailyMealMoneyConfig] = None,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> pd.DataFrame:
    """Build same-day 09:15 feature rows for every ticker/date."""
    config = config or DailyMealMoneyConfig()
    validate_daily_config(config)
    start_ts = pd.Timestamp(start_date) if start_date else None
    end_ts = pd.Timestamp(end_date) if end_date else None
    rows: list[dict] = []

    for ticker, df in intraday.items():
        if df.empty:
            continue
        work = df.copy()
        work["_date"] = pd.Index([pd.Timestamp(x).normalize().tz_localize(None) for x in work.index])
        grouped = work.groupby("_date", sort=True)
        daily = pd.DataFrame({
            "open": grouped["Open"].first(),
            "high": grouped["High"].max(),
            "low": grouped["Low"].min(),
            "close": grouped["Close"].last(),
            "volume": grouped["Volume"].sum(),
        })
        daily["prev_close"] = daily["close"].shift(1)
        daily["avg_volume_20d"] = daily["volume"].rolling(20).mean().shift(1)
        daily["prev_return_5d"] = daily["close"].shift(1) / daily["close"].shift(6) - 1

        for day in daily.index:
            if start_ts is not None and day < start_ts:
                continue
            if end_ts is not None and day > end_ts:
                continue
            prev_close = float(daily.at[day, "prev_close"])
            avg_volume = float(daily.at[day, "avg_volume_20d"])
            if not np.isfinite(prev_close) or not np.isfinite(avg_volume):
                continue
            if prev_close < config.min_price or prev_close > config.max_price:
                continue

            for (
                window_name,
                pre_start,
                pre_end,
                entry_time,
                force_exit_time,
                cutoff_time,
            ) in _trading_windows(config):
                pre = intraday_window(df, day, pre_start, pre_end)
                trade_bars = intraday_window(df, day, entry_time, force_exit_time)
                if pre.empty or trade_bars.empty:
                    continue
                pre_open = float(pre.iloc[0]["Open"])
                if not np.isfinite(pre_open) or pre_open <= 0:
                    continue

                entry_bar = trade_bars.iloc[0]
                long_edge = _long_edge_entry(float(entry_bar["Open"]), config)
                long_entry = (
                    long_edge if float(entry_bar["Low"]) <= long_edge
                    else float(entry_bar["Open"])
                )
                trade_bars = trade_bars.head(5)
                row = {
                    "Date": day,
                    "Ticker": ticker,
                    "Window": window_name,
                    "Window_End": cutoff_time,
                    "Pre_Start": pre_start,
                    "Pre_End": pre_end,
                    "Entry_Time": entry_time,
                    "Force_Exit_Time": force_exit_time,
                    "Prev_Close": prev_close,
                    "Avg_Volume_20D": avg_volume,
                    "Prev_Return_5D": float(daily.at[day, "prev_return_5d"])
                    if np.isfinite(daily.at[day, "prev_return_5d"]) else 0.0,
                    "Pre_Volume": float(pre["Volume"].sum()),
                    "Pre_Turnover": float(pre["Volume"].sum() * pre_open),
                    "Pre_Move_Pct": float(pre.iloc[-1]["Close"] / pre_open - 1),
                    "Pre_Range_Pct": float((pre["High"].max() - pre["Low"].min()) / pre_open),
                    "Rel_Pre_Volume": float(pre["Volume"].sum() / max(avg_volume, 1.0)),
                    "Entry_Price": float(long_entry),
                    "Open_Gap_Pct": float(long_entry / prev_close - 1),
                    "Exit_Close": float(trade_bars.iloc[-1]["Close"]),
                }
                for idx, (ts, bar) in enumerate(trade_bars.iterrows()):
                    row[f"Time_{idx}"] = pd.Timestamp(ts).strftime("%H:%M")
                    row[f"Open_{idx}"] = float(bar["Open"])
                    row[f"High_{idx}"] = float(bar["High"])
                    row[f"Low_{idx}"] = float(bar["Low"])
                    row[f"Close_{idx}"] = float(bar["Close"])
                for idx in range(len(trade_bars), 5):
                    row[f"Time_{idx}"] = ""
                    row[f"Open_{idx}"] = np.nan
                    row[f"High_{idx}"] = np.nan
                    row[f"Low_{idx}"] = np.nan
                    row[f"Close_{idx}"] = np.nan
                rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Date", "Ticker"]).reset_index(drop=True)


def _long_edge_entry(reference_price: float, config: DailyMealMoneyConfig) -> float:
    price = float(reference_price)
    for _ in range(max(config.entry_edge_ticks, 0)):
        price -= tick_size(price)
    return round_down_to_tick(price)


def _short_edge_entry(reference_price: float, config: DailyMealMoneyConfig) -> float:
    price = float(reference_price)
    for _ in range(max(config.entry_edge_ticks, 0)):
        price += tick_size(price)
    return round_up_to_tick(price)


def _score(frame: pd.DataFrame, side: str, config: DailyMealMoneyConfig) -> pd.Series:
    volume = np.log1p(frame["Pre_Turnover"].astype(float))
    rel_volume = frame["Rel_Pre_Volume"].astype(float).clip(lower=0, upper=1.2)
    price_range = frame["Pre_Range_Pct"].astype(float).clip(lower=0, upper=0.08)
    signed_move = (
        frame["Pre_Move_Pct"].astype(float)
        if side == "LONG" else -frame["Pre_Move_Pct"].astype(float)
    )
    abs_move = frame["Pre_Move_Pct"].astype(float).abs()
    raw_prev_return = (
        frame["Prev_Return_5D"].astype(float)
        if "Prev_Return_5D" in frame else pd.Series(0.0, index=frame.index)
    )
    prev_return = raw_prev_return if side == "LONG" else -raw_prev_return
    if config.score_mode == "volume_first":
        return volume * 4 + rel_volume * 40 + signed_move * 80 + price_range * 20
    if config.score_mode == "pressure":
        return volume * 2 + rel_volume * 25 + signed_move * 180 + price_range * 80
    if config.score_mode == "move_first":
        return volume + rel_volume * 15 + signed_move * 300 + price_range * 60
    if config.score_mode == "abs_pressure":
        return volume * 2 + rel_volume * 25 + abs_move * 120 + price_range * 90 + signed_move * 70
    if config.score_mode == "follow_prev":
        return volume * 2 + rel_volume * 25 + signed_move * 160 + price_range * 70 + prev_return * 40
    return volume * 2 + rel_volume * 20 + signed_move * 120 + price_range * 160


def select_daily_candidate_pool(features: pd.DataFrame,
                                config: Optional[DailyMealMoneyConfig] = None
                                ) -> pd.DataFrame:
    """Return all eligible long/short candidates sorted by daily priority."""
    config = config or DailyMealMoneyConfig()
    validate_daily_config(config)
    if features.empty:
        return pd.DataFrame()
    base = (
        (features["Avg_Volume_20D"] >= config.min_avg_volume)
        & (features["Pre_Volume"] >= config.min_pre_entry_volume)
        & (features["Pre_Range_Pct"] >= config.min_pre_entry_range_pct)
        & (features["Rel_Pre_Volume"] <= config.rel_pre_entry_volume_cap)
    )
    pieces = []
    if config.allow_long:
        long_frame = features[
            base
            & (features["Pre_Move_Pct"] >= config.min_abs_pre_entry_move_pct)
            & (features["Open_Gap_Pct"] >= config.long_gap_min)
            & (features["Open_Gap_Pct"] <= config.long_gap_max)
        ].copy()
        if not long_frame.empty:
            long_frame["Side"] = "LONG"
            long_frame["Score"] = _score(long_frame, "LONG", config)
            pieces.append(long_frame)
    if config.allow_short:
        short_frame = features[
            base
            & (features["Pre_Move_Pct"] <= -config.min_abs_pre_entry_move_pct)
            & (features["Open_Gap_Pct"] >= config.short_gap_min)
            & (features["Open_Gap_Pct"] <= config.short_gap_max)
        ].copy()
        if not short_frame.empty:
            short_frame["Side"] = "SHORT"
            short_frame["Score"] = (
                _score(short_frame, "SHORT", config) + config.short_score_bias
            )
            pieces.append(short_frame)
    if not pieces:
        return pd.DataFrame()
    candidates = pd.concat(pieces, ignore_index=True)
    if config.non_morning_window_score_penalty and "Window" in candidates:
        primary_window = _trading_windows(config)[0][0]
        candidates.loc[
            candidates["Window"] != primary_window, "Score"
        ] -= config.non_morning_window_score_penalty
    return (
        candidates
        .sort_values(["Date", "Score", "Pre_Turnover"], ascending=[True, False, False])
        .reset_index(drop=True)
    )


def select_daily_candidates(features: pd.DataFrame,
                            config: Optional[DailyMealMoneyConfig] = None
                            ) -> pd.DataFrame:
    """Select the top long/short candidate for each date by volume pressure."""
    candidates = select_daily_candidate_pool(features, config)
    if candidates.empty:
        return candidates
    return candidates.groupby("Date", as_index=False).head(1).reset_index(drop=True)


def simulate_daily_candidate(row: pd.Series,
                             config: Optional[DailyMealMoneyConfig] = None
                             ) -> Optional[dict]:
    """Simulate one selected long or short daily candidate."""
    config = config or DailyMealMoneyConfig()
    validate_daily_config(config)
    meal_config = config.as_meal_config()
    side = str(row["Side"])
    entry_time = str(row.get("Entry_Time", config.entry_time))
    force_exit_time = str(row.get("Force_Exit_Time", config.force_exit_time))
    window = str(row.get("Window", f"{entry_time}-{force_exit_time}"))
    window_end = str(row.get("Window_End", config.market_close_cutoff))
    if side == "LONG":
        entry_price = float(row["Entry_Price"])
    else:
        open0 = float(row["Open_0"])
        short_edge = _short_edge_entry(open0, config)
        entry_price = short_edge if float(row["High_0"]) >= short_edge else open0
    shares = calculate_shares(entry_price, meal_config)
    if shares <= 0:
        return None

    if side == "LONG":
        target_price = target_exit_price(entry_price, shares, meal_config)
        required_ticks = ticks_between(entry_price, target_price)
        stop_price = entry_price * (1 - config.stop_loss_pct)
    else:
        target_price = target_cover_price(entry_price, shares, config)
        required_ticks = short_ticks_between(entry_price, target_price)
        stop_price = entry_price * (1 + config.stop_loss_pct)
    if not np.isfinite(target_price) or required_ticks > config.max_required_ticks:
        return None

    exit_price = float(row["Exit_Close"])
    exit_time = force_exit_time
    reason = "TIME"
    for idx in range(5):
        low = float(row.get(f"Low_{idx}", np.nan))
        high = float(row.get(f"High_{idx}", np.nan))
        open_price = float(row.get(f"Open_{idx}", np.nan))
        bar_time = str(row.get(f"Time_{idx}", "")) or exit_time
        if not np.isfinite(low) or not np.isfinite(high) or not np.isfinite(open_price):
            continue
        if side == "LONG":
            if low <= stop_price:
                exit_price = open_price if open_price < stop_price else stop_price
                exit_time = bar_time
                reason = "SL"
                break
            if high >= target_price:
                exit_price = target_price
                exit_time = bar_time
                reason = "TARGET"
                break
        else:
            if high >= stop_price:
                exit_price = open_price if open_price > stop_price else stop_price
                exit_time = bar_time
                reason = "SL"
                break
            if low <= target_price:
                exit_price = target_price
                exit_time = bar_time
                reason = "TARGET"
                break

    pnl = (
        net_pnl(entry_price, exit_price, shares, meal_config)
        if side == "LONG"
        else short_net_pnl(entry_price, exit_price, shares, config)
    )
    return {
        "Date": pd.Timestamp(row["Date"]).strftime("%Y-%m-%d"),
        "Ticker": str(row["Ticker"]),
        "Window": window,
        "Window_End": window_end,
        "Side": side,
        "Entry_Time": entry_time,
        "Exit_Time": exit_time,
        "Entry_Price": round(float(entry_price), 4),
        "Exit_Price": round(float(exit_price), 4),
        "Shares": int(shares),
        "Gross_Capital": round(float(entry_price * shares), 2),
        "Pnl": round(float(pnl), 2),
        "Reason": reason,
        "Target_Price": round(float(target_price), 4),
        "Required_Ticks": int(required_ticks),
        "Score": round(float(row["Score"]), 4),
        "Pre_Volume": round(float(row["Pre_Volume"]), 0),
        "Pre_Move_Pct": round(float(row["Pre_Move_Pct"] * 100), 4),
        "Pre_Range_Pct": round(float(row["Pre_Range_Pct"] * 100), 4),
        "Open_Gap_Pct": round(float(row["Open_Gap_Pct"] * 100), 4),
        "Transaction_Tax_Rate": config.transaction_tax,
    }


def run_daily_backtest(intraday: Dict[str, pd.DataFrame],
                       config: Optional[DailyMealMoneyConfig] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None
                       ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the bidirectional daily strategy on local intraday bars."""
    config = config or DailyMealMoneyConfig()
    features = build_daily_features(intraday, config, start_date, end_date)
    trades = []
    if config.reselect_after_feasibility:
        candidates = select_daily_candidate_pool(features, config)
        for _, group in candidates.groupby("Date", sort=True):
            for _, row in group.iterrows():
                trade = simulate_daily_candidate(row, config)
                if trade is not None:
                    trades.append(trade)
                    break
    else:
        candidates = select_daily_candidates(features, config)
        for _, row in candidates.iterrows():
            trade = simulate_daily_candidate(row, config)
            if trade is not None:
                trades.append(trade)
    trades_df = pd.DataFrame(trades)
    daily_df = _daily_rows(features, trades_df, config)
    return trades_df, daily_df, summarize_daily(trades_df, daily_df, config)


def _daily_rows(features: pd.DataFrame, trades_df: pd.DataFrame,
                config: DailyMealMoneyConfig) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    dates = pd.Series(pd.to_datetime(features["Date"]).dt.strftime("%Y-%m-%d").unique())
    trade_map = (
        trades_df.set_index("Date")["Pnl"].to_dict()
        if not trades_df.empty else {}
    )
    rows = []
    for date in sorted(dates):
        pnl = float(trade_map.get(date, 0.0))
        rows.append({
            "Date": date,
            "Trades": 1 if date in trade_map else 0,
            "Pnl": round(pnl, 2),
            "Target_Achieved": bool(pnl >= config.target_min),
        })
    return pd.DataFrame(rows)


def summarize_daily(trades_df: pd.DataFrame, daily_df: pd.DataFrame,
                    config: DailyMealMoneyConfig) -> dict:
    active_days = int((daily_df["Trades"] > 0).sum()) if not daily_df.empty else 0
    total_trades = int(len(trades_df))
    target_trades = (
        int((trades_df["Reason"] == "TARGET").sum()) if not trades_df.empty else 0
    )
    win_trades = int((trades_df["Pnl"] > 0).sum()) if not trades_df.empty else 0
    long_trades = int((trades_df["Side"] == "LONG").sum()) if not trades_df.empty else 0
    if trades_df.empty:
        cutoff_violations = 0
    elif "Window_End" in trades_df:
        cutoff_violations = int((trades_df["Exit_Time"] >= trades_df["Window_End"]).sum())
    else:
        cutoff_violations = int((trades_df["Exit_Time"] >= config.market_close_cutoff).sum())
    return {
        "calendar_days": int(len(daily_df)),
        "active_days": active_days,
        "active_ratio": active_days / len(daily_df) if len(daily_df) else 0.0,
        "total_trades": total_trades,
        "target_trades": target_trades,
        "target_rate": target_trades / total_trades if total_trades else 0.0,
        "win_rate": win_trades / total_trades if total_trades else 0.0,
        "long_ratio": long_trades / total_trades if total_trades else 0.0,
        "total_pnl": round(float(trades_df["Pnl"].sum()), 2) if not trades_df.empty else 0.0,
        "avg_trade_pnl": round(float(trades_df["Pnl"].mean()), 2) if not trades_df.empty else 0.0,
        "cutoff_violations": cutoff_violations,
        "config": config.to_dict(),
    }


__all__ = [
    "DailyMealMoneyConfig",
    "DAYTRADE_TRANSACTION_TAX",
    "build_daily_features",
    "load_intraday_data",
    "read_intraday_csv",
    "run_daily_backtest",
    "select_daily_candidate_pool",
    "select_daily_candidates",
    "short_net_pnl",
    "target_cover_price",
    "simulate_daily_candidate",
    "summarize_daily",
]
