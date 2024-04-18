from datetime import datetime, timedelta
from time import sleep
import os

import pandas as pd
import vectorbt as vbt
import twstock

codes = twstock.codes


def get_past_x_days(stock_num, days=59):
    csv_path = f'./data/{stock_num}.csv'

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty and len(df) > 1:  # 檔案非空且超過一行
                print(f"Data for {stock_num} already exists and is not empty. Skipping download.")
                return
        except pd.errors.EmptyDataError:
            pass

    today = datetime.now()

    start_date = today - timedelta(days=days)

    end_date = today

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    yf_data = vbt.YFData.download(
        f"{stock_num}.TW",
        tz_convert='Asia/Taipei',
        tz_localize='Asia/Taipei',
        start=start_date_str + " 08:30:00 +0800",
        end=end_date_str + " 13:30:00 +0800",
        interval='5m',
        missing_index='raise',
        missing_columns='raise'
    )

    data = yf_data.get()
    data.to_csv(csv_path)


for k, v in codes.items():
    if v.market == '上市' and (v.type == '股票' or v.type == 'ETF'):
        get_past_x_days(k)
        print(v)
