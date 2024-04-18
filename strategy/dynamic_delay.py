import pandas as pd
import numpy as np
from datetime import timedelta

def trade(real_movement, delay=5, initial_state=1, initial_money=10000, max_buy=1, max_sell=1, print_log=True):
    """
    根據市場價格變動進行股票買賣模擬
    :param real_movement: 市場價格的真實變動序列 (假定此為DataFrame且包含時間戳索引)
    :param delay: 從買轉賣或賣轉買之間的延遲決策次數
    :param initial_state: 初始狀態，1 表示買，0 表示賣
    :param initial_money: 初始金額
    :param max_buy: 最大買入數量
    :param max_sell: 最大賣出數量
    :param print_log: 是否打印交易日誌
    """
    starting_money = initial_money
    state = initial_state
    current_inventory = 0
    states_buy, states_sell, states_entry, states_exit = [], [], [], []
    current_decision = 0
    real_movement = real_movement.close
    for i in range(1, real_movement.shape[0]):
        current_time = real_movement.index[i]
        current_price = real_movement.iloc[i]

        if state == 1 and current_price < real_movement.iloc[i - 1]:  # 考慮買入
            if current_decision >= delay:
                shares = min(initial_money // current_price, max_buy)
                if shares > 0:
                    initial_money -= shares * current_price
                    current_inventory += shares
                    states_buy.append(i)
                    if print_log:
                        print(f'{current_time}: buy {shares} units at price {current_price}, total balance {initial_money}')
                current_decision = 0
            else:
                current_decision += 1

        elif state == 0 and current_price > real_movement.iloc[i - 1]:  # 考慮賣出
            if current_decision >= delay:
                sell_units = min(current_inventory, max_sell)
                if sell_units > 0:
                    initial_money += sell_units * current_price
                    current_inventory -= sell_units
                    states_sell.append(i)
                    if print_log:
                        print(f'{current_time}: sell {sell_units} units at price {current_price}, total balance {initial_money}')
                current_decision = 0
            else:
                current_decision += 1

        state = 1 - state  # 切換狀態
        states_entry.append(state == 1)
        states_exit.append(state == 0)

    # 確保 states_entry 和 states_exit 的長度與 real_movement 相同
    if len(states_entry) < real_movement.shape[0]:
        states_entry.extend([False] * (real_movement.shape[0] - len(states_entry)))
    if len(states_exit) < real_movement.shape[0]:
        states_exit.extend([False] * (real_movement.shape[0] - len(states_exit)))

    total_gains = initial_money - starting_money
    invest = (total_gains / starting_money) * 100
    return states_buy, states_sell, states_entry, states_exit, total_gains, invest
