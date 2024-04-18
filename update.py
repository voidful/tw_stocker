from datetime import datetime, timedelta
from time import sleep
import os
import pandas as pd

import vectorbt as vbt
import twstock

codes = twstock.codes


def get_data_since_last_record(stock_num, base_path='./data/'):
    csv_path = f'{base_path}{stock_num}.csv'
    today = datetime.now()

    if os.path.exists(csv_path):
        data = pd.read_csv(csv_path, header=0)
        if not data.empty:
            try:
                last_record_date = pd.to_datetime(data['Datetime'].iloc[-1])
                start_date = last_record_date + timedelta(minutes=5)
            except:
                start_date = today - timedelta(days=59)
                pass
        else:
            start_date = today - timedelta(days=59)

    else:
        start_date = today - timedelta(days=59)

    end_date = today

    start_date_str = start_date.strftime('%Y-%m-%d %H:%M:%S +0800')
    end_date_str = end_date.strftime('%Y-%m-%d %H:%M:%S +0800')

    yf_data = vbt.YFData.download(
        f"{stock_num}.TW",
        tz_convert='Asia/Taipei',
        tz_localize='Asia/Taipei',
        start=start_date_str,
        end=end_date_str,
        interval='5m',
        missing_index='drop'
    )

    new_data = yf_data.get()

    if os.path.exists(csv_path):
        new_data.to_csv(csv_path, mode='a', header=False)
    else:
        new_data.to_csv(csv_path)

    return new_data


for k, v in codes.items():
    if v.market == '上市' and (v.type == '股票' or v.type == 'ETF'):
        new_data = get_data_since_last_record(k)
        print(f"Updated data for {k}")
