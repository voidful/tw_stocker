import pandas as pd

from strategy.meal_money import (
    MealMoneyConfig,
    calculate_shares,
    net_pnl,
    rank_candidates,
    simulate_trade,
    target_exit_price,
    tick_size,
    ticks_between,
    trade_cost_breakdown,
)


def make_bars(rows):
    idx = pd.to_datetime([f"2026-01-02 {time}+08:00" for time, *_ in rows])
    return pd.DataFrame(
        {
            "Open": [row[1] for row in rows],
            "High": [row[2] for row in rows],
            "Low": [row[3] for row in rows],
            "Close": [row[4] for row in rows],
            "Volume": [1000 for _ in rows],
        },
        index=idx,
    )


def test_fee_adjusted_target_exit_price_reaches_net_target():
    config = MealMoneyConfig(target_min=500, slippage=0)
    entry = 140.0
    shares = calculate_shares(entry, config)

    exit_price = target_exit_price(entry, shares, config)
    pnl = net_pnl(entry, exit_price, shares, config)

    assert 15_000 <= shares * entry <= 100_000
    assert tick_size(entry) == 0.5
    assert pnl >= 500.00


def test_production_defaults_encode_validated_constraints():
    config = MealMoneyConfig()

    assert config.target_min == 500.0
    assert config.min_trade_capital == 15_000.0
    assert config.trade_capital == 100_000.0
    assert config.max_trade_capital == 100_000.0
    assert config.min_price == 103.0
    assert config.max_price == 180.0
    assert config.min_avg_volume == 2_000_000
    assert config.min_entry_bar_volume == 0
    assert config.min_pre_entry_volume == 2_000_000
    assert config.min_pre_entry_move_pct == 0.020
    assert config.min_pre_entry_range_pct == 0.015
    assert config.use_market_filter is False
    assert config.liquidity_rank_weight == 0.5
    assert config.market_follow_rank_weight == 0.2
    assert config.entry_time == "09:15"
    assert config.force_exit_time == "09:35"
    assert config.min_open_gap_pct == 0.005
    assert config.max_open_gap_pct == 0.040
    assert config.max_required_ticks == 3
    assert config.stop_loss_pct == 0.005
    assert config.allowed_sectors == ()
    assert config.buy_cost == 0.001425
    assert config.sell_cost == 0.004425
    assert config.transaction_tax == 0.003
    assert config.min_commission == 20.0
    assert config.lot_size == 1
    assert config.slippage == 0.0


def test_small_capital_cost_breakdown_uses_min_commission_and_tax():
    config = MealMoneyConfig()
    costs = trade_cost_breakdown(150.0, 150.0, 100, config)

    assert costs["entry_gross"] == 15_000
    assert costs["buy_commission"] == 21.375
    assert costs["sell_commission"] == 21.375
    assert costs["transaction_tax"] == 45.0
    assert costs["round_trip_cost"] == 87.75


def test_liquidity_rank_can_lift_high_volume_candidate():
    config = MealMoneyConfig(
        score_rank_weight=1.0,
        liquidity_rank_weight=2.0,
        market_follow_rank_weight=0.0,
        top_k_candidates=1,
    )
    candidates = [
        {"ticker": "A", "score": 3.0, "avg_volume_20d": 1_000_000, "prev_return_5d": 0.01},
        {"ticker": "B", "score": 2.8, "avg_volume_20d": 20_000_000, "prev_return_5d": 0.01},
    ]

    ranked = rank_candidates(candidates, config)

    assert ranked[0]["ticker"] == "B"
    assert ranked[0]["liquidity_rank"] > ranked[0]["score_rank"]


def test_simulate_trade_exits_at_target_before_940():
    config = MealMoneyConfig(target_min=500, slippage=0)
    entry = 140.0
    shares = calculate_shares(entry, config)
    tp = target_exit_price(entry, shares, config)
    bars = make_bars([
        ("09:05", entry, entry + 0.10, entry - 0.05, entry + 0.05),
        ("09:10", entry + 0.05, tp + 0.01, entry, tp),
        ("09:35", tp, tp, tp, tp),
    ])

    trade = simulate_trade("1234", "2026-01-02", bars, entry, shares, 3.0, 0.005, config)

    assert trade["Reason"] == "TARGET"
    assert trade["Exit_Time"] == "09:10"
    assert trade["Exit_Time"] < "09:40"
    assert trade["Pnl"] >= 500
    assert trade["Required_Ticks"] == ticks_between(entry, tp)


def test_same_bar_stop_takes_priority_over_target():
    config = MealMoneyConfig(target_min=500, stop_loss_pct=0.02)
    entry = 140.0
    shares = calculate_shares(entry, config)
    tp = target_exit_price(entry, shares, config)
    bars = make_bars([
        ("09:05", entry, tp + 0.50, entry * 0.98, entry),
    ])

    trade = simulate_trade("1234", "2026-01-02", bars, entry, shares, 3.0, 0.005, config)

    assert trade["Reason"] == "SL"
    assert trade["Pnl"] < 0


def test_force_exit_uses_935_close_when_target_not_hit():
    config = MealMoneyConfig(target_min=500)
    entry = 140.0
    shares = calculate_shares(entry, config)
    bars = make_bars([
        ("09:05", entry, entry + 0.05, entry - 0.05, entry + 0.02),
        ("09:20", entry + 0.02, entry + 0.06, entry - 0.02, entry + 0.03),
        ("09:35", entry + 0.03, entry + 0.07, entry, entry + 0.04),
    ])

    trade = simulate_trade("1234", "2026-01-02", bars, entry, shares, 3.0, 0.005, config)

    assert trade["Reason"] == "TIME"
    assert trade["Exit_Time"] == "09:35"
    assert trade["Exit_Time"] < "09:40"
