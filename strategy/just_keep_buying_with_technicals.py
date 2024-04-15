def trade(real_movement, adx, rsi, atr, investment_amount_base,
          adx_threshold=10, rsi_lower_bound=45, rsi_upper_bound=65, atr_threshold=0.1,
          investment_interval=1, fee_rate=0.001425, slippage=0.0005,
          initial_money=10000, print_log=True):
    """
    持續買進策略 (基於技術指標優化)

    參數:
        real_movement: 市場價格的實際變動序列
        adx: 平均趨向指數 (ADX) 指標序列
        rsi: 相對強弱指數 (RSI) 指標序列
        atr: 平均真實波幅 (ATR) 指標序列
        investment_amount_base: 基礎投資金額
        adx_threshold: ADX 趨勢判斷閾值
        rsi_lower_bound: RSI 超賣閾值
        rsi_upper_bound: RSI 超買閾值
        atr_threshold: ATR 波動性閾值
        investment_interval: 投資間隔 (以天數為單位)
        fee_rate: 交易手續費率
        slippage: 滑點率
        initial_money: 初始資金
        print_log: 是否列印交易日誌

    返回:
        states_buy, states_sell, states_entry, states_exit, total_gains, invest
    """

    current_money = initial_money
    shares = 0
    states_buy, states_sell, states_entry, states_exit = [], [], [], []

    for i, price in enumerate(real_movement):
        # 判斷是否達到投資時間點
        if i % investment_interval == 0 and current_money >= investment_amount_base:
            # 判斷市場趨勢 (根據 ADX 指標)
            trend = "up" if adx[i] > adx_threshold else "down_or_sideways"
            print(f"Trend: {trend}")
            # 判斷市場動量 (根據 RSI 指標)
            if rsi[i] < rsi_lower_bound:
                momentum = "oversold"
            elif rsi[i] > rsi_upper_bound:
                momentum = "overbought"
            else:
                momentum = "neutral"

            # 判斷市場波動性 (根據 ATR 指標)
            volatility = "high" if atr[i] > atr_threshold else "low"
            print(f"Volatility: {volatility}")
            # 根據指標綜合判斷是否買入
            if trend == "up" and momentum != "overbought" and volatility != "high":
                # 計算投資金額
                investment_amount = investment_amount_base

                if momentum == "oversold":
                    investment_amount *= 1.5  # 超賣狀態，增加投資金額

                # 計算交易費用
                fee = investment_amount * fee_rate

                # 計算滑點後的實際買入價格
                actual_price = price * (1 + slippage)

                # 計算可買入股數
                buy_shares = (investment_amount - fee) // actual_price

                # 更新資金和持股數量
                current_money -= buy_shares * actual_price + fee
                shares += buy_shares
                states_buy.append(i)
                if print_log:
                    print(f'slot {i}: buy {buy_shares} units at price {actual_price}, total balance {current_money}')

    total_value = shares * real_movement[-1] + current_money
    total_gains = total_value - initial_money
    invest = (total_gains / initial_money) * 100
    return states_buy, states_sell, states_entry, states_exit, total_gains, invest
