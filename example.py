import vectorbt as vbt
import pandas as pd
import numpy as np
import fta
from strategy.dynamic_delay import trade

url = "https://raw.githubusercontent.com/voidful/tw_stocker/main/data/2330.csv"
df = pd.read_csv(url, index_col='Datetime')


ta = fta.TA_Features()
df_full = ta.get_all_indicators(df)

PARAMETER = {
    "delay": 54,
    "initial_money": 10000,
    "max_buy": 10,
    "max_sell": 10,
}

states_buy, states_sell, states_entry, states_exit, total_gains, invest = trade(df_full.close, **PARAMETER)

fees = 0 # 假設交易費用為 0
portfolio_kwargs = dict(size=np.inf, fees=float(fees), freq='5m')
portfolio = vbt.Portfolio.from_signals(df_full['close'], states_entry, states_exit, **portfolio_kwargs)
print(portfolio.stats())
portfolio.plot().show()