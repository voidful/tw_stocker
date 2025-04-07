from datetime import datetime, timedelta
import os
from time import sleep

import pandas as pd
import pytz
import vectorbt as vbt
import twstock

codes = twstock.codes

def clean_csv(csv_path):
    """Clean the CSV file by removing rows with incorrect number of columns."""
    with open(csv_path, 'r') as file:
        lines = file.readlines()

    num_columns = len(lines[0].split(','))

    with open(csv_path, 'w') as file:
        for line in lines:
            if len(line.split(',')) == num_columns:
                file.write(line)

def get_data_since_last_record(stock_num, base_path='./data/'):
    csv_path = f'{base_path}{stock_num}.csv'
    tz_taipei = pytz.timezone('Asia/Taipei')
    today = datetime.now(tz_taipei).replace(hour=0, minute=0, second=0, microsecond=0)

    # 預設開始時間為過去 59 天
    start_date = today - timedelta(days=59)

    if os.path.exists(csv_path):
        try:
            clean_csv(csv_path)
            data = pd.read_csv(csv_path, header=0)
        except pd.errors.ParserError as e:
            print(f"[{stock_num}] CSV 解析錯誤: {e}")
            data = pd.DataFrame()

        if not data.empty:
            try:
                last_record_date = pd.to_datetime(data['Datetime'].iloc[-1]).tz_convert('Asia/Taipei')
                start_date = last_record_date + timedelta(minutes=5)
            except Exception as e:
                print(f"[{stock_num}] 時間欄位轉換錯誤: {e}")
                # fallback 已處理，保持 start_date 為預設值

    end_date = today + timedelta(hours=14)

    try:
        yf_data = vbt.YFData.download(
            f"{stock_num}.TW",
            start=start_date.strftime('%Y-%m-%d %H:%M:%S'),
            end=end_date.strftime('%Y-%m-%d %H:%M:%S'),
            interval='5m',
            missing_index='drop'
        )
        new_data = yf_data.get()

        if new_data.empty:
            print(f"[{stock_num}] 無資料下載")
            return pd.DataFrame()

        # 儲存資料
        if os.path.exists(csv_path):
            new_data.to_csv(csv_path, mode='a', header=False)
        else:
            new_data.to_csv(csv_path)

        sleep(2)  # 緩衝避免 rate limit
        return new_data

    except Exception as e:
        print(f"[{stock_num}] 下載錯誤: {e}")
        return pd.DataFrame()

# --- 主程式 ---
for k, v in codes.items():
    if v.market == '上市' and (v.type == '股票' or v.type == 'ETF'):
        print(f"處理中: {k} - {v.name}")
        new_data = get_data_since_last_record(k)
        if new_data.empty:
            print(f"[{k}] 無新資料或下載失敗")
        else:
            print(f"[{k}] 資料已更新，共 {len(new_data)} 筆")
