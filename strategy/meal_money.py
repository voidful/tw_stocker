"""Morning scalping strategy for a small fixed daily profit target.

The strategy reuses the v8.5 overnight setup idea, but the final intraday
candidate order is liquidity-first: no sector is excluded by default, and the
watchlist follows high-volume stocks in a healthy market tape.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from strategy.sector_flow import SECTOR_MAP, classify_sector


COMMISSION_RATE = 0.001425
STOCK_TRANSACTION_TAX_RATE = 0.003
MIN_COMMISSION = 20.0
BUY_COST_RATE = COMMISSION_RATE
DAYTRADE_TAX_RATE = STOCK_TRANSACTION_TAX_RATE
DAYTRADE_SELL_COST_RATE = COMMISSION_RATE + STOCK_TRANSACTION_TAX_RATE
SLIPPAGE_RATE = 0.001


@dataclass(frozen=True)
class MealMoneyConfig:
    """Configuration for the meal-money morning strategy."""

    target_min: float = 500.0
    target_max: float = 800.0
    min_trade_capital: float = 15_000.0
    trade_capital: float = 100_000.0
    max_trade_capital: float = 100_000.0
    max_trades_per_day: int = 1
    top_n_universe: int = 60
    top_k_candidates: int = 7
    threshold: float = 2.0
    min_price: float = 103.0
    max_price: float = 180.0
    min_avg_volume: float = 2_000_000.0
    min_entry_bar_volume: float = 0.0
    pre_entry_start_time: str = "09:00"
    pre_entry_end_time: str = "09:10"
    min_pre_entry_volume: float = 2_000_000.0
    min_pre_entry_move_pct: float = 0.020
    min_pre_entry_range_pct: float = 0.015
    use_market_filter: bool = False
    min_market_breadth_20: float = 0.40
    min_market_return_5d: float = -0.020
    score_rank_weight: float = 1.0
    liquidity_rank_weight: float = 0.5
    market_follow_rank_weight: float = 0.2
    entry_time: str = "09:15"
    force_exit_time: str = "09:35"
    market_close_cutoff: str = "09:40"
    min_open_gap_pct: float = 0.005
    max_open_gap_pct: float = 0.040
    max_required_move_pct: float = 0.045
    max_required_ticks: int = 3
    entry_edge_ticks: int = 1
    stop_loss_pct: float = 0.005
    buy_cost: float = COMMISSION_RATE
    sell_cost: float = COMMISSION_RATE + STOCK_TRANSACTION_TAX_RATE
    transaction_tax: float = STOCK_TRANSACTION_TAX_RATE
    min_commission: float = MIN_COMMISSION
    slippage: float = 0.0
    lot_size: int = 1
    use_sox_filter: bool = False
    min_sox_1d: float = -0.02
    min_sox_5d: float = -0.05
    allowed_tech_gates: tuple[str, ...] = ("open", "strong", "boost")
    allowed_sectors: tuple[str, ...] = ()
    use_night_filter: bool = False
    min_night_return_pct: float = -0.4
    focus_times: tuple[str, ...] = (
        "09:15", "10:00", "11:00", "11:40", "12:00", "12:50", "13:00", "13:20"
    )

    def to_dict(self) -> dict:
        return asdict(self)


def validate_config(config: MealMoneyConfig) -> None:
    if config.target_min <= 0:
        raise ValueError("target_min must be positive")
    if config.target_max < config.target_min:
        raise ValueError("target_max must be greater than or equal to target_min")
    if config.min_trade_capital < 14_000:
        raise ValueError("min_trade_capital must be at least 14000")
    if config.trade_capital < config.min_trade_capital:
        raise ValueError("trade_capital must be at least min_trade_capital")
    if config.max_trade_capital < config.min_trade_capital:
        raise ValueError("max_trade_capital must be at least min_trade_capital")
    if config.trade_capital > config.max_trade_capital:
        raise ValueError("trade_capital must not exceed max_trade_capital")
    if config.max_trades_per_day < 1:
        raise ValueError("max_trades_per_day must be at least 1")
    if config.lot_size < 1:
        raise ValueError("lot_size must be at least 1")
    if config.force_exit_time >= config.market_close_cutoff:
        raise ValueError("force_exit_time must be before market_close_cutoff")
    if config.entry_time < "09:15":
        raise ValueError("entry_time must not be before 09:15")
    if config.pre_entry_start_time >= config.entry_time:
        raise ValueError("pre_entry_start_time must be before entry_time")
    if config.pre_entry_end_time >= config.entry_time:
        raise ValueError("pre_entry_end_time must be before entry_time")
    if not (0 < config.stop_loss_pct < 1):
        raise ValueError("stop_loss_pct must be between 0 and 1")
    if config.min_price <= 0 or config.max_price <= config.min_price:
        raise ValueError("price band must be positive and ordered")
    if config.max_required_ticks < 1:
        raise ValueError("max_required_ticks must be at least 1")
    if config.buy_cost < 0 or config.sell_cost < 0 or config.transaction_tax < 0:
        raise ValueError("cost rates must be non-negative")
    if config.min_commission < 0:
        raise ValueError("min_commission must be non-negative")
    if not (0 <= config.min_market_breadth_20 <= 1):
        raise ValueError("min_market_breadth_20 must be between 0 and 1")
    if min(
        config.score_rank_weight,
        config.liquidity_rank_weight,
        config.market_follow_rank_weight,
    ) < 0:
        raise ValueError("selection rank weights must be non-negative")
    if min(
        config.min_pre_entry_volume,
        config.min_pre_entry_move_pct,
        config.min_pre_entry_range_pct,
    ) < 0:
        raise ValueError("pre-entry filters must be non-negative")


def calculate_shares(entry_price: float, config: MealMoneyConfig) -> int:
    """Return shares sized by configured capital while preserving the floor."""
    if not np.isfinite(entry_price) or entry_price <= 0:
        return 0
    gross_unit = entry_price * (1 + config.slippage)
    raw_shares = int(config.trade_capital // gross_unit)
    shares = (raw_shares // config.lot_size) * config.lot_size
    while shares > 0 and entry_cash_cost(entry_price, shares, config) > config.max_trade_capital:
        shares -= config.lot_size
    if shares <= 0 or shares * entry_price < config.min_trade_capital:
        return 0
    return shares


def tick_size(price: float) -> float:
    """TWSE tick size for normal listed stock prices."""
    if price < 5:
        return 0.01
    if price < 10:
        return 0.05
    if price < 50:
        return 0.1
    if price < 100:
        return 0.5
    if price < 500:
        return 0.5
    if price < 1000:
        return 1.0
    return 5.0


def round_up_to_tick(price: float) -> float:
    step = tick_size(price)
    return float(np.ceil((price - 1e-12) / step) * step)


def round_down_to_tick(price: float) -> float:
    step = tick_size(price)
    return float(np.floor((price + 1e-12) / step) * step)


def ticks_between(entry_price: float, exit_price: float) -> int:
    step = tick_size(entry_price)
    return int(np.ceil(max(exit_price - entry_price, 0) / step - 1e-12))


def _sell_commission_rate(config: MealMoneyConfig) -> float:
    return max(config.sell_cost - config.transaction_tax, 0.0)


def buy_commission(entry_price: float, shares: int, config: MealMoneyConfig) -> float:
    gross = shares * entry_price
    return max(gross * config.buy_cost, config.min_commission)


def sell_commission(exit_price: float, shares: int, config: MealMoneyConfig) -> float:
    gross = shares * exit_price
    return max(gross * _sell_commission_rate(config), config.min_commission)


def transaction_tax(exit_price: float, shares: int, config: MealMoneyConfig) -> float:
    return shares * exit_price * config.transaction_tax


def entry_cash_cost(entry_price: float, shares: int, config: MealMoneyConfig) -> float:
    gross = shares * entry_price
    return gross + buy_commission(entry_price, shares, config) + gross * config.slippage


def trade_cost_breakdown(entry_price: float, exit_price: float, shares: int,
                         config: MealMoneyConfig) -> dict[str, float]:
    entry_gross = shares * entry_price
    exit_gross = shares * exit_price
    buy_fee = buy_commission(entry_price, shares, config)
    sell_fee = sell_commission(exit_price, shares, config)
    tax = transaction_tax(exit_price, shares, config)
    slippage_cost = (entry_gross + exit_gross) * config.slippage
    return {
        "entry_gross": entry_gross,
        "exit_gross": exit_gross,
        "buy_commission": buy_fee,
        "sell_commission": sell_fee,
        "transaction_tax": tax,
        "slippage_cost": slippage_cost,
        "round_trip_cost": buy_fee + sell_fee + tax + slippage_cost,
        "entry_cash_cost": entry_gross + buy_fee + entry_gross * config.slippage,
    }


def net_pnl(entry_price: float, exit_price: float, shares: int,
            config: MealMoneyConfig) -> float:
    """Net profit after buy fee, sell fee, tax model, and slippage."""
    entry_gross = shares * entry_price
    exit_gross = shares * exit_price
    cost = entry_gross + buy_commission(entry_price, shares, config) + entry_gross * config.slippage
    proceeds = (
        exit_gross
        - sell_commission(exit_price, shares, config)
        - transaction_tax(exit_price, shares, config)
        - exit_gross * config.slippage
    )
    return proceeds - cost


def target_exit_price(entry_price: float, shares: int,
                      config: MealMoneyConfig) -> float:
    """Raw exit price required to reach the configured net daily target."""
    if shares <= 0:
        return np.nan
    entry_cost = entry_cash_cost(entry_price, shares, config)
    denominator = shares * (
        1 - _sell_commission_rate(config) - config.transaction_tax - config.slippage
    )
    if denominator <= 0:
        return np.nan
    estimate = (entry_cost + config.target_min + config.min_commission) / denominator
    price = round_up_to_tick(max(entry_price, estimate))
    for _ in range(1000):
        if net_pnl(entry_price, price, shares, config) >= config.target_min:
            return price
        price = round_up_to_tick(price + tick_size(price))
    return np.nan


def one_tick_edge_entry(reference_price: float, config: MealMoneyConfig) -> float:
    """Entry price one or more ticks below the reference price."""
    price = float(reference_price)
    for _ in range(max(config.entry_edge_ticks, 0)):
        price -= tick_size(price)
    return round_down_to_tick(price)


def read_intraday_csv(path: str | Path) -> pd.DataFrame:
    """Read one local 5-minute OHLCV CSV."""
    df = pd.read_csv(path)
    if "Datetime" not in df.columns:
        raise ValueError(f"{path} does not contain a Datetime column")
    df["Datetime"] = (
        pd.to_datetime(df["Datetime"], errors="coerce", utc=True)
        .dt.tz_convert("Asia/Taipei")
    )
    df = df.dropna(subset=["Datetime"]).set_index("Datetime").sort_index()
    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[needed].dropna(subset=["Open", "High", "Low", "Close"])


def load_intraday_data(data_dir: str | Path = "data",
                       tickers: Optional[Iterable[str]] = None) -> Dict[str, pd.DataFrame]:
    """Load local ticker CSV files from data_dir."""
    data_dir = Path(data_dir)
    ticker_set = {str(t) for t in tickers} if tickers else None
    paths = sorted(data_dir.glob("*.csv"))
    out: Dict[str, pd.DataFrame] = {}
    for path in paths:
        ticker = path.stem
        if ticker_set is not None and ticker not in ticker_set:
            continue
        try:
            df = read_intraday_csv(path)
        except Exception:
            continue
        if not df.empty:
            out[ticker] = df
    return out


def _date_index(index: pd.Index) -> pd.Index:
    return pd.Index([pd.Timestamp(x).date() for x in index])


def _time_labels(index: pd.Index) -> pd.Index:
    return pd.Index([pd.Timestamp(x).strftime("%H:%M") for x in index])


def aggregate_daily(intraday: Dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Aggregate local 5-minute bars into daily panels."""
    fields = {key: {} for key in ["open", "high", "low", "close", "volume"]}
    for ticker, df in intraday.items():
        work = df.copy()
        work["_date"] = _date_index(work.index)
        grouped = work.groupby("_date", sort=True)
        daily = pd.DataFrame({
            "open": grouped["Open"].first(),
            "high": grouped["High"].max(),
            "low": grouped["Low"].min(),
            "close": grouped["Close"].last(),
            "volume": grouped["Volume"].sum(),
        })
        for field in fields:
            fields[field][ticker] = daily[field]

    panels = {}
    for field, series_map in fields.items():
        panel = pd.DataFrame(series_map)
        panel.index = pd.to_datetime(panel.index)
        panels[field] = panel.sort_index()
    return panels


def build_v85_like_scores(close_df: pd.DataFrame, vol_df: pd.DataFrame,
                          top_n: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mirror the production score: rank_momentum(20d)*3 + rank_trend(60MA)."""
    turnover = (close_df * vol_df).rolling(20).mean()
    universe = turnover.rank(axis=1, ascending=False) <= top_n

    mom_20 = close_df / close_df.shift(20)
    ma_60 = close_df.rolling(60).mean()
    trend_bias = close_df / ma_60

    mom_rank = mom_20.where(universe).rank(axis=1, pct=True)
    trend_rank = trend_bias.where(universe).rank(axis=1, pct=True)
    score = (mom_rank * 3 + trend_rank).where(universe)
    return score, ma_60


def load_night_market_csv(path: str | Path | None) -> pd.DataFrame:
    """Load optional night-session context.

    Expected columns:
    - Date: Taiwan regular-session date to which the night data applies.
    - Night_Return_Pct: percent return, for example -0.35 for -0.35%.
    """
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError("night market CSV must contain Date")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    return df


def enrich_us_signals(us_signals: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Add simple SOX return fields used by the morning strategy."""
    if us_signals is None or us_signals.empty:
        return pd.DataFrame()
    out = us_signals.copy()
    if "sox_close" in out.columns:
        out["sox_ret_1d"] = out["sox_close"].pct_change()
        out["sox_ret_5d"] = out["sox_close"].pct_change(5)
    return out


def market_context_allows_entry(day: pd.Timestamp,
                                us_signals: Optional[pd.DataFrame],
                                night_market: Optional[pd.DataFrame],
                                config: MealMoneyConfig) -> tuple[bool, dict]:
    """Check SOX and night-session gates for the Taiwan session."""
    context = {
        "Sox_Ret_1D": np.nan,
        "Sox_Ret_5D": np.nan,
        "Tech_Gate": "",
        "Night_Return_Pct": np.nan,
    }
    if us_signals is not None and not us_signals.empty:
        eligible = us_signals.loc[us_signals.index < day]
        if not eligible.empty:
            row = eligible.iloc[-1]
            sox_1d = float(row.get("sox_ret_1d", np.nan))
            sox_5d = float(row.get("sox_ret_5d", np.nan))
            tech_gate = str(row.get("tech_gate", ""))
            context.update({
                "Sox_Ret_1D": sox_1d,
                "Sox_Ret_5D": sox_5d,
                "Tech_Gate": tech_gate,
            })

    if config.use_sox_filter:
        if us_signals is None or us_signals.empty:
            return False, context
        if not np.isfinite(context["Sox_Ret_1D"]) or not np.isfinite(context["Sox_Ret_5D"]):
            return False, context
        if context["Sox_Ret_1D"] < config.min_sox_1d or context["Sox_Ret_5D"] < config.min_sox_5d:
            return False, context
        if context["Tech_Gate"] and context["Tech_Gate"] not in config.allowed_tech_gates:
            return False, context

    day_key = pd.Timestamp(day).normalize()
    if night_market is not None and not night_market.empty and day_key in night_market.index:
        night_ret = float(night_market.at[day_key, "Night_Return_Pct"])
        context["Night_Return_Pct"] = night_ret

    if config.use_night_filter:
        if night_market is None or night_market.empty:
            return False, context
        if not np.isfinite(context["Night_Return_Pct"]):
            return False, context
        if context["Night_Return_Pct"] < config.min_night_return_pct:
            return False, context

    return True, context


def local_market_context_allows_entry(prev_day: pd.Timestamp,
                                      context: dict,
                                      config: MealMoneyConfig) -> tuple[bool, dict]:
    """Use only completed local-market data to decide whether the tape is tradable."""
    breadth = _series_value(context.get("market_breadth_20"), prev_day)
    market_ret_1d = _series_value(context.get("market_return_1d"), prev_day)
    market_ret_5d = _series_value(context.get("market_return_5d"), prev_day)
    out = {
        "Market_Breadth_20": breadth,
        "Market_Return_1D": market_ret_1d,
        "Market_Return_5D": market_ret_5d,
    }
    if not config.use_market_filter:
        return True, out
    if not np.isfinite(breadth) or breadth < config.min_market_breadth_20:
        return False, out
    if not np.isfinite(market_ret_5d) or market_ret_5d < config.min_market_return_5d:
        return False, out
    return True, out


def _series_value(series: object, key: pd.Timestamp) -> float:
    if not isinstance(series, pd.Series) or key not in series.index:
        return np.nan
    try:
        value = float(series.at[key])
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def intraday_window(df: pd.DataFrame, day: pd.Timestamp | str,
                    start_time: str, end_time: str) -> pd.DataFrame:
    """Return bars for one date between start_time and end_time, inclusive."""
    day_str = pd.Timestamp(day).strftime("%Y-%m-%d")
    start = pd.Timestamp(f"{day_str} {start_time}")
    end = pd.Timestamp(f"{day_str} {end_time}")
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        start = start.tz_localize(df.index.tz)
        end = end.tz_localize(df.index.tz)
    return df.loc[start:end].copy()


def simulate_trade(ticker: str, day: pd.Timestamp | str, bars: pd.DataFrame,
                   entry_price: float, shares: int, score: float,
                   open_gap_pct: float, config: MealMoneyConfig) -> dict:
    """Simulate one morning trade with conservative same-bar ordering."""
    if bars.empty:
        raise ValueError("bars cannot be empty")
    tp_price = target_exit_price(entry_price, shares, config)
    stop_price = entry_price * (1 - config.stop_loss_pct)
    exit_price = float(bars.iloc[-1]["Close"])
    exit_time = pd.Timestamp(bars.index[-1]).strftime("%H:%M")
    reason = "TIME"

    for ts, row in bars.iterrows():
        bar_time = pd.Timestamp(ts).strftime("%H:%M")
        low = float(row["Low"])
        high = float(row["High"])
        open_price = float(row["Open"])

        if low <= stop_price:
            exit_price = open_price if open_price < stop_price else stop_price
            exit_time = bar_time
            reason = "SL"
            break
        if high >= tp_price:
            exit_price = tp_price
            exit_time = bar_time
            reason = "TARGET"
            break

    pnl = net_pnl(entry_price, exit_price, shares, config)
    gross_capital = entry_price * shares
    costs = trade_cost_breakdown(entry_price, exit_price, shares, config)
    return {
        "Date": pd.Timestamp(day).strftime("%Y-%m-%d"),
        "Ticker": ticker,
        "Sector": classify_sector(ticker),
        "Sector_Label": SECTOR_MAP.get(classify_sector(ticker), {}).get("label", "其他"),
        "Entry_Time": pd.Timestamp(bars.index[0]).strftime("%H:%M"),
        "Exit_Time": exit_time,
        "Entry_Price": round(float(entry_price), 4),
        "Exit_Price": round(float(exit_price), 4),
        "Shares": int(shares),
        "Gross_Capital": round(float(gross_capital), 2),
        "Entry_Cash_Cost": round(float(costs["entry_cash_cost"]), 2),
        "Pnl": round(float(pnl), 2),
        "Pnl_Pct": round(float(pnl / gross_capital * 100), 4) if gross_capital else np.nan,
        "Buy_Commission": round(float(costs["buy_commission"]), 2),
        "Sell_Commission": round(float(costs["sell_commission"]), 2),
        "Transaction_Tax": round(float(costs["transaction_tax"]), 2),
        "Slippage_Cost": round(float(costs["slippage_cost"]), 2),
        "Round_Trip_Cost": round(float(costs["round_trip_cost"]), 2),
        "Reason": reason,
        "Score": round(float(score), 4),
        "Open_Gap_Pct": round(float(open_gap_pct * 100), 4),
        "Required_Move_Pct": round(float((tp_price / entry_price - 1) * 100), 4),
        "Required_Ticks": ticks_between(entry_price, tp_price),
        "Tick_Size": tick_size(entry_price),
        "Target_Exit_Price": round(float(tp_price), 4),
        "Target_Min": round(float(config.target_min), 2),
    }


def rank_candidates(candidates: list[dict],
                    config: MealMoneyConfig) -> list[dict]:
    """Rank candidates by liquidity first, then momentum and market following."""
    if not candidates:
        return []
    frame = pd.DataFrame({
        "score": [c["score"] for c in candidates],
        "avg_volume_20d": [c["avg_volume_20d"] for c in candidates],
        "prev_return_5d": [c["prev_return_5d"] for c in candidates],
    })
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if frame[col].notna().any():
            frame[col] = frame[col].fillna(frame[col].median())
        else:
            frame[col] = frame[col].fillna(0.0)

    score_rank = frame["score"].rank(pct=True)
    liquidity_rank = frame["avg_volume_20d"].rank(pct=True)
    follow_rank = frame["prev_return_5d"].rank(pct=True)
    selection_score = (
        score_rank * config.score_rank_weight
        + liquidity_rank * config.liquidity_rank_weight
        + follow_rank * config.market_follow_rank_weight
    )
    for idx, candidate in enumerate(candidates):
        candidate["score_rank"] = float(score_rank.iloc[idx])
        candidate["liquidity_rank"] = float(liquidity_rank.iloc[idx])
        candidate["market_follow_rank"] = float(follow_rank.iloc[idx])
        candidate["selection_score"] = float(selection_score.iloc[idx])
    return sorted(
        candidates,
        key=lambda c: (
            c["selection_score"],
            c["avg_volume_20d"],
            c["score"],
        ),
        reverse=True,
    )[:config.top_k_candidates]


def select_candidates_for_day(intraday: Dict[str, pd.DataFrame],
                              score_df: pd.DataFrame,
                              ma_df: pd.DataFrame,
                              close_df: pd.DataFrame,
                              avg_vol_20: pd.DataFrame,
                              ret_5d_df: pd.DataFrame,
                              day: pd.Timestamp,
                              prev_day: pd.Timestamp,
                              config: MealMoneyConfig) -> list[dict]:
    """Build eligible same-day candidates using only previous-day data."""
    if prev_day not in score_df.index:
        return []
    prev_scores = score_df.loc[prev_day].dropna().sort_values(ascending=False)
    candidates = []

    for ticker, score in prev_scores.items():
        if score < config.threshold or ticker not in intraday:
            continue
        sector = classify_sector(ticker)
        if config.allowed_sectors and sector not in config.allowed_sectors:
            continue
        prev_close = close_df.at[prev_day, ticker] if ticker in close_df.columns else np.nan
        prev_ma = ma_df.at[prev_day, ticker] if ticker in ma_df.columns else np.nan
        if not np.isfinite(prev_close) or not np.isfinite(prev_ma) or prev_close <= prev_ma:
            continue
        if prev_close < config.min_price or prev_close > config.max_price:
            continue
        avg_volume = avg_vol_20.at[prev_day, ticker] if ticker in avg_vol_20.columns else np.nan
        if not np.isfinite(avg_volume) or avg_volume < config.min_avg_volume:
            continue

        pre_bars = intraday_window(
            intraday[ticker], day, config.pre_entry_start_time, config.pre_entry_end_time
        )
        if pre_bars.empty:
            continue
        pre_open = float(pre_bars.iloc[0]["Open"])
        if not np.isfinite(pre_open) or pre_open <= 0:
            continue
        pre_volume = float(pre_bars["Volume"].sum())
        pre_move_pct = float(pre_bars.iloc[-1]["Close"] / pre_open - 1)
        pre_range_pct = float((pre_bars["High"].max() - pre_bars["Low"].min()) / pre_open)
        if pre_volume < config.min_pre_entry_volume:
            continue
        if pre_move_pct < config.min_pre_entry_move_pct:
            continue
        if pre_range_pct < config.min_pre_entry_range_pct:
            continue

        bars = intraday_window(
            intraday[ticker], day, config.entry_time, config.force_exit_time
        )
        if bars.empty:
            continue
        entry_bar = bars.iloc[0]
        entry_bar_volume = float(entry_bar.get("Volume", 0))
        if entry_bar_volume < config.min_entry_bar_volume:
            continue
        edge_entry = one_tick_edge_entry(float(entry_bar["Open"]), config)
        entry_price = edge_entry if float(entry_bar["Low"]) <= edge_entry else float(entry_bar["Open"])
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue
        open_gap_pct = entry_price / float(prev_close) - 1
        if open_gap_pct < config.min_open_gap_pct:
            continue
        if open_gap_pct > config.max_open_gap_pct:
            continue

        shares = calculate_shares(entry_price, config)
        if shares <= 0:
            continue
        tp_price = target_exit_price(entry_price, shares, config)
        required_move_pct = tp_price / entry_price - 1
        if required_move_pct > config.max_required_move_pct:
            continue
        if ticks_between(entry_price, tp_price) > config.max_required_ticks:
            continue
        prev_return_5d = (
            ret_5d_df.at[prev_day, ticker] if ticker in ret_5d_df.columns else np.nan
        )

        candidates.append({
            "ticker": ticker,
            "score": float(score),
            "bars": bars,
            "entry_price": entry_price,
            "shares": shares,
            "open_gap_pct": float(open_gap_pct),
            "required_move_pct": float(required_move_pct),
            "avg_volume_20d": float(avg_volume),
            "entry_bar_volume": float(entry_bar_volume),
            "prev_return_5d": float(prev_return_5d) if np.isfinite(prev_return_5d) else np.nan,
            "pre_entry_volume": float(pre_volume),
            "pre_entry_move_pct": float(pre_move_pct),
            "pre_entry_range_pct": float(pre_range_pct),
        })
    return rank_candidates(candidates, config)


def run_backtest(intraday: Dict[str, pd.DataFrame],
                 config: Optional[MealMoneyConfig] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 us_signals: Optional[pd.DataFrame] = None,
                 night_market: Optional[pd.DataFrame] = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Backtest the meal-money strategy on local 5-minute bars."""
    config = config or MealMoneyConfig()
    validate_config(config)
    if not intraday:
        raise ValueError("No intraday data loaded")

    context = prepare_backtest_context(intraday, config, us_signals)
    return run_backtest_from_context(
        intraday, context, config,
        start_date=start_date, end_date=end_date, night_market=night_market,
    )


def prepare_backtest_context(intraday: Dict[str, pd.DataFrame],
                             config: MealMoneyConfig,
                             us_signals: Optional[pd.DataFrame] = None) -> dict:
    """Precompute data shared by repeated Meal Money experiments."""
    panels = aggregate_daily(intraday)
    close_df = panels["close"]
    vol_df = panels["volume"]
    score_df, ma_df = build_v85_like_scores(close_df, vol_df, config.top_n_universe)
    avg_vol_20 = vol_df.rolling(20).mean()
    ma_20 = close_df.rolling(20).mean()
    valid_count = close_df.notna().sum(axis=1).replace(0, np.nan)
    market_breadth_20 = (close_df > ma_20).sum(axis=1) / valid_count
    market_return_1d = close_df.pct_change().median(axis=1)
    market_return_5d = close_df.pct_change(5).median(axis=1)
    ret_5d_df = close_df.pct_change(5)
    us_signals = enrich_us_signals(us_signals)
    return {
        "panels": panels,
        "close_df": close_df,
        "vol_df": vol_df,
        "score_df": score_df,
        "ma_df": ma_df,
        "avg_vol_20": avg_vol_20,
        "ret_5d_df": ret_5d_df,
        "market_breadth_20": market_breadth_20,
        "market_return_1d": market_return_1d,
        "market_return_5d": market_return_5d,
        "us_signals": us_signals,
        "dates": list(close_df.index),
    }


def run_backtest_from_context(intraday: Dict[str, pd.DataFrame],
                              context: dict,
                              config: MealMoneyConfig,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None,
                              night_market: Optional[pd.DataFrame] = None
                              ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run one config from precomputed Meal Money context."""
    validate_config(config)
    close_df = context["close_df"]
    score_df = context["score_df"]
    ma_df = context["ma_df"]
    avg_vol_20 = context["avg_vol_20"]
    ret_5d_df = context["ret_5d_df"]
    us_signals = context.get("us_signals", pd.DataFrame())
    dates = context["dates"]
    trades = []
    daily_rows = []
    start_ts = pd.Timestamp(start_date) if start_date else None
    end_ts = pd.Timestamp(end_date) if end_date else None

    for idx in range(1, len(dates)):
        day = dates[idx]
        prev_day = dates[idx - 1]
        if start_ts is not None and day < start_ts:
            continue
        if end_ts is not None and day > end_ts:
            continue
        local_context_ok, local_market_context = local_market_context_allows_entry(
            prev_day, context, config
        )
        context_ok, market_context = market_context_allows_entry(day, us_signals, night_market, config)
        market_context = {**local_market_context, **market_context}
        if not local_context_ok:
            context_ok = False
        if not context_ok:
            daily_rows.append({
                "Date": day.strftime("%Y-%m-%d"),
                "Candidates": 0,
                "Trades": 0,
                "Pnl": 0.0,
                "Target_Achieved": False,
                "Blocked_By_Context": True,
                **market_context,
            })
            continue

        candidates = select_candidates_for_day(
            intraday, score_df, ma_df, close_df, avg_vol_20, ret_5d_df,
            day, prev_day, config
        )
        day_pnl = 0.0
        day_trades = 0
        for candidate in candidates:
            if day_trades >= config.max_trades_per_day:
                break
            if day_pnl >= config.target_min:
                break
            trade = simulate_trade(
                candidate["ticker"],
                day,
                candidate["bars"],
                candidate["entry_price"],
                candidate["shares"],
                candidate["score"],
                candidate["open_gap_pct"],
                config,
            )
            trade.update({
                "Avg_Volume_20D": round(float(candidate["avg_volume_20d"]), 0),
                "Entry_Bar_Volume": round(float(candidate["entry_bar_volume"]), 0),
                "Pre_Entry_Volume": round(float(candidate["pre_entry_volume"]), 0),
                "Pre_Entry_Move_Pct": round(float(candidate["pre_entry_move_pct"] * 100), 4),
                "Pre_Entry_Range_Pct": round(float(candidate["pre_entry_range_pct"] * 100), 4),
                "Prev_Return_5D": round(float(candidate["prev_return_5d"] * 100), 4)
                if np.isfinite(candidate["prev_return_5d"]) else np.nan,
                "Selection_Score": round(float(candidate["selection_score"]), 4),
                "Score_Rank": round(float(candidate["score_rank"]), 4),
                "Liquidity_Rank": round(float(candidate["liquidity_rank"]), 4),
                "Market_Follow_Rank": round(float(candidate["market_follow_rank"]), 4),
                **market_context,
            })
            trades.append(trade)
            day_pnl += trade["Pnl"]
            day_trades += 1

        daily_rows.append({
            "Date": day.strftime("%Y-%m-%d"),
            "Candidates": len(candidates),
            "Trades": day_trades,
            "Pnl": round(float(day_pnl), 2),
            "Target_Achieved": bool(day_pnl >= config.target_min),
            "Blocked_By_Context": False,
            **market_context,
        })

    trades_df = pd.DataFrame(trades)
    daily_df = pd.DataFrame(daily_rows)
    summary = summarize(trades_df, daily_df, config)
    return trades_df, daily_df, summary


def summarize(trades_df: pd.DataFrame, daily_df: pd.DataFrame,
              config: MealMoneyConfig) -> dict:
    """Compute strategy-level metrics."""
    active_days = int((daily_df["Trades"] > 0).sum()) if not daily_df.empty else 0
    success_days = int(daily_df["Target_Achieved"].sum()) if not daily_df.empty else 0
    total_trades = int(len(trades_df))
    target_trades = int((trades_df["Reason"] == "TARGET").sum()) if not trades_df.empty else 0
    win_trades = int((trades_df["Pnl"] > 0).sum()) if not trades_df.empty else 0
    cutoff_violations = (
        int((trades_df["Exit_Time"] >= config.market_close_cutoff).sum())
        if not trades_df.empty else 0
    )
    success_trade_count = 0
    if not trades_df.empty and not daily_df.empty:
        success_dates = set(daily_df.loc[daily_df["Target_Achieved"], "Date"])
        success_trade_count = int(trades_df["Date"].isin(success_dates).sum())

    return {
        "calendar_days": int(len(daily_df)),
        "active_days": active_days,
        "success_days": success_days,
        "daily_success_rate": success_days / active_days if active_days else 0.0,
        "total_trades": total_trades,
        "target_trades": target_trades,
        "trade_target_rate": target_trades / total_trades if total_trades else 0.0,
        "win_rate": win_trades / total_trades if total_trades else 0.0,
        "avg_trades_per_active_day": total_trades / active_days if active_days else 0.0,
        "avg_trades_per_success_day": (
            success_trade_count / success_days if success_days else 0.0
        ),
        "total_pnl": float(trades_df["Pnl"].sum()) if not trades_df.empty else 0.0,
        "avg_trade_pnl": float(trades_df["Pnl"].mean()) if not trades_df.empty else 0.0,
        "avg_day_pnl": float(daily_df["Pnl"].mean()) if not daily_df.empty else 0.0,
        "cutoff_violations": cutoff_violations,
        "config": config.to_dict(),
    }


def summarize_by_sector(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Long-term sector tracking table."""
    if trades_df.empty:
        return pd.DataFrame()
    grouped = trades_df.groupby(["Sector", "Sector_Label"], dropna=False)
    summary = grouped.agg(
        Trades=("Pnl", "size"),
        Total_Pnl=("Pnl", "sum"),
        Avg_Pnl=("Pnl", "mean"),
        Win_Rate=("Pnl", lambda s: float((s > 0).mean())),
        Target_Rate=("Reason", lambda s: float((s == "TARGET").mean())),
        Avg_Required_Ticks=("Required_Ticks", "mean"),
    ).reset_index()
    return summary.sort_values(["Total_Pnl", "Trades"], ascending=[False, False])


def compute_time_focus_stats(intraday: Dict[str, pd.DataFrame],
                             config: MealMoneyConfig) -> pd.DataFrame:
    """Track which intraday focus times actually have exploitable movement."""
    rows = []
    for ticker, df in intraday.items():
        sector = classify_sector(ticker)
        work = df.copy()
        work["_time"] = _time_labels(work.index)
        for focus_time in config.focus_times:
            sample = work[work["_time"] == focus_time]
            if sample.empty:
                continue
            open_price = sample["Open"].replace(0, np.nan)
            range_pct = (sample["High"] - sample["Low"]) / open_price
            close_move_pct = (sample["Close"] - sample["Open"]) / open_price
            rows.append({
                "Ticker": ticker,
                "Sector": sector,
                "Sector_Label": SECTOR_MAP.get(sector, {}).get("label", "其他"),
                "Focus_Time": focus_time,
                "Samples": int(range_pct.dropna().shape[0]),
                "Avg_Range_Pct": float(range_pct.mean() * 100),
                "Median_Range_Pct": float(range_pct.median() * 100),
                "Avg_Abs_Move_Pct": float(close_move_pct.abs().mean() * 100),
                "Avg_Volume": float(sample["Volume"].mean()),
            })
    if not rows:
        return pd.DataFrame()
    stats = pd.DataFrame(rows)
    return stats.sort_values(["Avg_Range_Pct", "Avg_Volume"], ascending=[False, False])


def build_latest_watchlist(intraday: Dict[str, pd.DataFrame],
                           config: Optional[MealMoneyConfig] = None) -> pd.DataFrame:
    """Build a next-session watchlist from the latest completed day."""
    config = config or MealMoneyConfig()
    validate_config(config)
    context = prepare_backtest_context(intraday, config)
    close_df = context["close_df"]
    score_df = context["score_df"]
    ma_df = context["ma_df"]
    avg_vol_20 = context["avg_vol_20"]
    ret_5d_df = context["ret_5d_df"]
    if close_df.empty:
        return pd.DataFrame()
    signal_day = close_df.index[-1]
    context_ok, market_context = local_market_context_allows_entry(signal_day, context, config)
    if not context_ok:
        return pd.DataFrame([{
            "Signal_Date": signal_day.strftime("%Y-%m-%d"),
            "Ticker": "",
            "Status": "Blocked_By_Local_Market_Context",
            **market_context,
        }]).iloc[0:0]
    rows = []
    candidates = []
    scores = score_df.loc[signal_day].dropna().sort_values(ascending=False)
    for ticker, score in scores.items():
        if score < config.threshold:
            continue
        prev_close = close_df.at[signal_day, ticker]
        prev_ma = ma_df.at[signal_day, ticker] if ticker in ma_df.columns else np.nan
        if not np.isfinite(prev_close) or not np.isfinite(prev_ma) or prev_close <= prev_ma:
            continue
        if prev_close < config.min_price or prev_close > config.max_price:
            continue
        avg_volume = avg_vol_20.at[signal_day, ticker] if ticker in avg_vol_20.columns else np.nan
        if not np.isfinite(avg_volume) or avg_volume < config.min_avg_volume:
            continue
        estimate_shares = calculate_shares(prev_close, config)
        if estimate_shares <= 0:
            continue
        estimate_target = target_exit_price(prev_close, estimate_shares, config)
        if not np.isfinite(estimate_target):
            continue
        if estimate_target / prev_close - 1 > config.max_required_move_pct:
            continue
        if ticks_between(prev_close, estimate_target) > config.max_required_ticks:
            continue
        prev_return_5d = ret_5d_df.at[signal_day, ticker] if ticker in ret_5d_df.columns else np.nan
        candidates.append({
            "ticker": ticker,
            "score": float(score),
            "avg_volume_20d": float(avg_volume),
            "prev_return_5d": float(prev_return_5d) if np.isfinite(prev_return_5d) else np.nan,
            "prev_close": float(prev_close),
            "estimate_shares": int(estimate_shares),
            "estimate_target": float(estimate_target),
        })

    for candidate in rank_candidates(candidates, config):
        ticker = candidate["ticker"]
        prev_close = candidate["prev_close"]
        estimate_shares = candidate["estimate_shares"]
        estimate_target = candidate["estimate_target"]
        sector = classify_sector(ticker)
        rows.append({
            "Signal_Date": signal_day.strftime("%Y-%m-%d"),
            "Ticker": ticker,
            "Sector": sector,
            "Sector_Label": SECTOR_MAP.get(sector, {}).get("label", "其他"),
            "Score": round(float(candidate["score"]), 4),
            "Selection_Score": round(float(candidate["selection_score"]), 4),
            "Score_Rank": round(float(candidate["score_rank"]), 4),
            "Liquidity_Rank": round(float(candidate["liquidity_rank"]), 4),
            "Market_Follow_Rank": round(float(candidate["market_follow_rank"]), 4),
            "Reference_Close": round(float(prev_close), 4),
            "One_Tick_Edge_Bid": one_tick_edge_entry(prev_close, config),
            "Valid_Entry_Min": round_up_to_tick(float(prev_close * (1 + config.min_open_gap_pct))),
            "Valid_Entry_Max": round_down_to_tick(float(prev_close * (1 + config.max_open_gap_pct))),
            "Estimated_Shares": int(estimate_shares),
            "Estimated_Gross_Capital": round(float(estimate_shares * prev_close), 2),
            "Estimated_Target_Exit": round(float(estimate_target), 4),
            "Required_Ticks": ticks_between(prev_close, estimate_target),
            "Avg_Volume_20D": round(float(candidate["avg_volume_20d"]), 0),
            "Prev_Return_5D": round(float(candidate["prev_return_5d"] * 100), 4)
            if np.isfinite(candidate["prev_return_5d"]) else np.nan,
            "Pre_Entry_Window": f"{config.pre_entry_start_time}-{config.pre_entry_end_time}",
            "Min_Pre_Entry_Volume": round(float(config.min_pre_entry_volume), 0),
            "Min_Pre_Entry_Move_Pct": round(float(config.min_pre_entry_move_pct * 100), 4),
            "Min_Pre_Entry_Range_Pct": round(float(config.min_pre_entry_range_pct * 100), 4),
            **market_context,
            "Exit_Deadline": config.force_exit_time,
            "Target_Net_Min": config.target_min,
            "Target_Net_Max": config.target_max,
        })
    return pd.DataFrame(rows)
