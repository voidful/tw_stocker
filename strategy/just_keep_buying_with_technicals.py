import pandas as pd
import numpy as np

def trade(real_movement,
          adx_threshold=10, rsi_lower_bound=45, rsi_upper_bound=65, atr_threshold=0.1,
          investment_interval=1, fee_rate=0.001425, slippage=0.0005,
          initial_money=10000, print_log=False):
    """
    持續買進策略 (基於技術指標優化)
    """

    current_money = initial_money
    shares = 0
    states_buy, states_sell = [], []
    states_entry = [False] * len(real_movement)  # 初始所有時間點沒有交易入場
    states_exit = [False] * len(real_movement)   # 初始所有時間點沒有交易出場

    for i in range(len(real_movement)):
        current_time = real_movement.index[i]
        # 使用 .iloc 進行基於位置的索引
        current_price = real_movement.iloc[i]['close']
        adx = real_movement.iloc[i]['adx']
        rsi = real_movement.iloc[i]['rsi']
        atr = real_movement.iloc[i]['atr']

        if i % investment_interval == 0 and current_money >= initial_money:
            trend = "up" if adx > adx_threshold else "down_or_sideways"
            momentum = "oversold" if rsi < rsi_lower_bound else ("overbought" if rsi > rsi_upper_bound else "neutral")
            volatility = "high" if atr > atr_threshold else "low"

            if trend == "up" and momentum != "overbought" and volatility != "high":
                investment_amount = initial_money * (1.5 if momentum == "oversold" else 1)
                fee = investment_amount * fee_rate
                actual_price = current_price * (1 + slippage)
                buy_shares = (investment_amount - fee) // actual_price

                current_money -= buy_shares * actual_price + fee
                shares += buy_shares
                states_buy.append(i)
                states_entry[i] = True
                if print_log:
                    print(f'{current_time}: buy {buy_shares} units at price {actual_price}, total balance {current_money}')

    total_value = shares * real_movement['close'].iloc[-1] + current_money
    total_gains = total_value - initial_money
    invest = (total_gains / initial_money) * 100
    return states_buy, states_sell, states_entry, states_exit, total_gains, invest
