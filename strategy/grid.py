import pandas as pd
import numpy as np

def trade(real_movement, initial_money=10000, rsi_period=14, low_rsi=30, high_rsi=70, ema_period=26, print_log=False):
    money = initial_money
    states_buy = []
    states_sell = []
    states_entry = [False] * len(real_movement)  # 初始化所有條目為False
    states_exit = [False] * len(real_movement)   # 初始化所有條目為False
    current_inventory = 0

    # 計算RSI
    delta = real_movement['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    real_movement['rsi'] = 100 - (100 / (1 + rs))

    # 計算EMA
    real_movement['ema'] = real_movement['close'].ewm(span=ema_period, adjust=False).mean()

    # 計算布林帶
    real_movement['ma20'] = real_movement['close'].rolling(window=20).mean()
    real_movement['stddev'] = real_movement['close'].rolling(window=20).std()
    real_movement['upper_band'] = real_movement['ma20'] + (real_movement['stddev'] * 2)
    real_movement['lower_band'] = real_movement['ma20'] - (real_movement['stddev'] * 2)

    def buy(i, price):
        nonlocal money, current_inventory
        if money >= price:
            shares = 1  # 或者其他計算股數的邏輯
            current_inventory += shares
            money -= price * shares
            states_buy.append(i)
            states_entry[i] = True
            if print_log:
                print(f'{real_movement.index[i]}: buy {shares} units at price {price}, total balance {money}')
        else:
            if print_log:
                print(f'{real_movement.index[i]}: attempted to buy, insufficient funds.')

    def sell(i, price):
        nonlocal money, current_inventory
        if current_inventory > 0:
            shares = min(current_inventory, 1)  # 或者其他計算股數的邏輯
            current_inventory -= shares
            money += price * shares
            states_sell.append(i)
            states_exit[i] = True
            if print_log:
                print(f'{real_movement.index[i]}: sell {shares} units at price {price}, total balance {money}')
        else:
            if print_log:
                print(f'{real_movement.index[i]}: attempted to sell, no inventory.')

    for i in range(len(real_movement)):
        if real_movement['rsi'].iloc[i] < low_rsi and real_movement['close'].iloc[i] < real_movement['lower_band'].iloc[i]:
            buy(i, real_movement['close'].iloc[i])
        elif real_movement['rsi'].iloc[i] > high_rsi and real_movement['close'].iloc[i] > real_movement['upper_band'].iloc[i]:
            sell(i, real_movement['close'].iloc[i])

    invest = ((money - initial_money) / initial_money) * 100
    total_gains = money + current_inventory * real_movement['close'].iloc[-1] - initial_money

    return states_buy, states_sell, states_entry, states_exit, total_gains, invest
