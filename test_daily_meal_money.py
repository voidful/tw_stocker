from argparse import Namespace

import numpy as np
import pandas as pd

from daily_meal_money_report import apply_preset, make_config
from strategy.daily_meal_money import (
    DEFAULT_MEAL_TRADING_WINDOWS,
    DailyMealMoneyConfig,
    select_daily_candidates,
    short_net_pnl,
    simulate_daily_candidate,
    target_cover_price,
)
from strategy.meal_money import calculate_shares


def test_daily_defaults_encode_bidirectional_lunch_strategy():
    config = DailyMealMoneyConfig()

    assert config.target_min == 600.0
    assert config.trade_capital == 100_000.0
    assert config.min_price == 103.0
    assert config.max_price == 180.0
    assert config.min_avg_volume == 500_000
    assert config.min_pre_entry_volume == 1_000_000
    assert config.min_abs_pre_entry_move_pct == 0.0
    assert config.min_pre_entry_range_pct == 0.015
    assert config.transaction_tax == 0.0015
    assert config.stop_loss_pct == 0.020
    assert config.max_required_ticks == 4
    assert config.allow_long is True
    assert config.allow_short is True
    assert config.score_mode == "abs_pressure"
    assert config.short_score_bias == 30.0
    assert config.reselect_after_feasibility is False


def test_small_cap_daily_preset_favors_near_daily_low_cap_trading():
    args = Namespace(
        preset="small-cap-daily",
        target_min=600.0,
        min_trade_capital=15_000.0,
        trade_capital=100_000.0,
        max_trade_capital=100_000.0,
        min_price=103.0,
        max_price=180.0,
        min_avg_volume=500_000.0,
        min_pre_entry_volume=1_000_000.0,
        min_abs_pre_entry_move_pct=0.0,
        min_pre_entry_range_pct=0.015,
        rel_pre_entry_volume_cap=99.0,
        long_gap_min=-0.015,
        long_gap_max=0.025,
        short_gap_min=-0.025,
        short_gap_max=0.015,
        max_required_ticks=4,
        entry_edge_ticks=1,
        stop_loss_pct=0.020,
        transaction_tax=0.0015,
        buy_cost=0.001425,
        sell_commission=0.001425,
        min_commission=20.0,
        slippage=0.0,
        lot_size=1,
        score_mode="abs_pressure",
        short_score_bias=30.0,
        reselect_after_feasibility=False,
        use_meal_windows=False,
        non_morning_window_score_penalty=0.0,
        long_only=False,
        short_only=False,
    )

    config = make_config(apply_preset(args))

    assert config.target_min == 420.0
    assert config.trade_capital == 20_000.0
    assert config.max_trade_capital == 20_000.0
    assert config.min_avg_volume == 1_000_000
    assert config.min_pre_entry_volume == 500_000
    assert config.min_pre_entry_range_pct == 0.012
    assert config.rel_pre_entry_volume_cap == 2.0
    assert config.max_required_ticks == 6
    assert config.stop_loss_pct == 0.020
    assert config.allow_long is True
    assert config.allow_short is True
    assert config.transaction_tax == 0.0015
    assert config.score_mode == "abs_pressure"
    assert config.short_score_bias == 10.0
    assert config.reselect_after_feasibility is True
    assert config.trading_windows == DEFAULT_MEAL_TRADING_WINDOWS
    assert len(config.trading_windows) == 5
    assert config.non_morning_window_score_penalty == 20.0


def test_simulate_candidate_uses_row_specific_trading_window_times():
    config = DailyMealMoneyConfig(target_min=500, stop_loss_pct=0.01)
    row = pd.Series({
        "Date": pd.Timestamp("2026-01-02"),
        "Ticker": "1234",
        "Window": "10:00-10:20",
        "Window_End": "10:20",
        "Entry_Time": "10:10",
        "Force_Exit_Time": "10:15",
        "Side": "SHORT",
        "Score": 100.0,
        "Pre_Volume": 900_000,
        "Pre_Move_Pct": -0.01,
        "Pre_Range_Pct": 0.012,
        "Open_Gap_Pct": -0.006,
        "Exit_Close": 138.0,
        "Time_0": "10:10",
        "Open_0": 140.0,
        "High_0": 140.5,
        "Low_0": 139.5,
        "Close_0": 139.5,
        "Time_1": "10:15",
        "Open_1": 139.0,
        "High_1": 139.2,
        "Low_1": 130.0,
        "Close_1": 138.0,
    })
    for idx in range(2, 5):
        row[f"Time_{idx}"] = ""
        row[f"Open_{idx}"] = np.nan
        row[f"High_{idx}"] = np.nan
        row[f"Low_{idx}"] = np.nan
        row[f"Close_{idx}"] = np.nan

    trade = simulate_daily_candidate(row, config)

    assert trade["Window"] == "10:00-10:20"
    assert trade["Entry_Time"] == "10:10"
    assert trade["Exit_Time"] == "10:15"
    assert trade["Exit_Time"] < trade["Window_End"]
    assert trade["Reason"] == "TARGET"


def test_short_target_cover_price_reaches_net_target():
    config = DailyMealMoneyConfig(target_min=500)
    entry = 140.0
    shares = calculate_shares(entry, config.as_meal_config())

    cover = target_cover_price(entry, shares, config)
    pnl = short_net_pnl(entry, cover, shares, config)

    assert cover < entry
    assert pnl >= 500


def test_simulate_short_candidate_hits_target_before_940():
    config = DailyMealMoneyConfig(target_min=500, stop_loss_pct=0.01)
    row = pd.Series({
        "Date": pd.Timestamp("2026-01-02"),
        "Ticker": "1234",
        "Side": "SHORT",
        "Score": 100.0,
        "Pre_Volume": 900_000,
        "Pre_Move_Pct": -0.01,
        "Pre_Range_Pct": 0.012,
        "Open_Gap_Pct": -0.006,
        "Exit_Close": 138.0,
        "Time_0": "09:15",
        "Open_0": 140.0,
        "High_0": 140.5,
        "Low_0": 130.0,
        "Close_0": 138.0,
    })
    for idx in range(1, 5):
        row[f"Time_{idx}"] = ""
        row[f"Open_{idx}"] = np.nan
        row[f"High_{idx}"] = np.nan
        row[f"Low_{idx}"] = np.nan
        row[f"Close_{idx}"] = np.nan

    trade = simulate_daily_candidate(row, config)

    assert trade["Side"] == "SHORT"
    assert trade["Reason"] == "TARGET"
    assert trade["Exit_Time"] == "09:15"
    assert trade["Exit_Time"] < "09:40"
    assert trade["Pnl"] >= 500


def test_select_daily_candidates_can_choose_short_by_pressure():
    config = DailyMealMoneyConfig(
        min_avg_volume=2_000_000,
        min_pre_entry_volume=500_000,
        min_abs_pre_entry_move_pct=0.003,
        min_pre_entry_range_pct=0.003,
        score_mode="pressure",
        short_score_bias=0.0,
    )
    features = pd.DataFrame([
        {
            "Date": pd.Timestamp("2026-01-02"),
            "Ticker": "LONG",
            "Avg_Volume_20D": 3_000_000,
            "Pre_Volume": 700_000,
            "Pre_Turnover": 84_000_000,
            "Pre_Move_Pct": 0.004,
            "Pre_Range_Pct": 0.004,
            "Rel_Pre_Volume": 0.23,
            "Open_Gap_Pct": 0.004,
        },
        {
            "Date": pd.Timestamp("2026-01-02"),
            "Ticker": "SHORT",
            "Avg_Volume_20D": 3_000_000,
            "Pre_Volume": 1_400_000,
            "Pre_Turnover": 168_000_000,
            "Pre_Move_Pct": -0.008,
            "Pre_Range_Pct": 0.012,
            "Rel_Pre_Volume": 0.47,
            "Open_Gap_Pct": -0.006,
        },
    ])

    candidates = select_daily_candidates(features, config)

    assert len(candidates) == 1
    assert candidates.iloc[0]["Ticker"] == "SHORT"
    assert candidates.iloc[0]["Side"] == "SHORT"
