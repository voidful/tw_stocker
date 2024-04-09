def trade(real_movement, delay=5, initial_state=1, initial_money=10000, max_buy=1, max_sell=1, print_log=True):
    """
    根據市場價格變動進行股票買賣模擬
    :param real_movement: 市場價格的真實變動序列
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

    for i in range(real_movement.shape[0]):
        if state == 1 and real_movement.iloc[i] < real_movement.iloc[i - 1]:  # 考慮買入
            if current_decision >= delay:
                shares = min(initial_money // real_movement.iloc[i], max_buy)
                if shares > 0:
                    initial_money -= shares * real_movement.iloc[i]
                    current_inventory += shares
                    states_buy.append(i)
                    if print_log:
                        print(
                            f'slot {i}: buy {shares} units at price {real_movement.iloc[i]}, total balance {initial_money}')
                current_decision = 0
            else:
                current_decision += 1

        elif state == 0 and real_movement.iloc[i] > real_movement.iloc[i - 1]:  # 考慮賣出
            if current_decision >= delay:
                sell_units = min(current_inventory, max_sell)
                if sell_units > 0:
                    initial_money += sell_units * real_movement.iloc[i]
                    current_inventory -= sell_units
                    states_sell.append(i)
                    if print_log:
                        print(
                            f'slot {i}: sell {sell_units} units at price {real_movement.iloc[i]}, total balance {initial_money}')
                current_decision = 0
            else:
                current_decision += 1

        state = 1 - state  # 切換狀態
        states_entry.append(state == 1)
        states_exit.append(state == 0)

    total_gains = initial_money - starting_money
    invest = (total_gains / starting_money) * 100
    return states_buy, states_sell, states_entry, states_exit, total_gains, invest
