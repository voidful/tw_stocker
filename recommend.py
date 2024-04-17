# 按照每天爬的data和dynamic delay策略來推薦買賣股票
import nlp2
import pandas as pd
import fta
from tqdm import tqdm

from strategy.just_keep_buying_with_technicals import trade


def recommend_stock(url, parameters):
    # 爬取每天的股票數據
    df = pd.read_csv(url, index_col='Datetime')

    df.columns = map(str.lower, df.columns)
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

    # 使用dynamic delay策略來決定是否買賣股票
    states_buy, states_sell, states_entry, states_exit, total_gains, invest = trade(df.close, **parameters)

    # 取得今天的數據
    today = df.index[-1]
    today_close_price = df.close.iloc[-1]

    # 判斷今天是否應該買賣
    should_buy = today in states_buy
    should_sell = today in states_sell

    # 返回今天是否應該買賣以及買賣推薦的價格
    return should_buy, should_sell, today_close_price


for i in tqdm(nlp2.get_files_from_dir("data")):
    try:
        url = i
        parameters = {
            "delay": 54,
            "initial_money": 10000,
            "max_buy": 10,
            "max_sell": 10,
            "print_log": False
        }
        should_buy, should_sell, today_close_price = recommend_stock(url, parameters)
        if should_sell or should_buy:
            print(f"{i.split('/')[-1].split('.')[0]} Should buy today: {should_buy}, Should sell today: {should_sell}, Recommended price: {today_close_price}")
    except Exception as e:
        pass
