from datetime import datetime, timedelta
import os
from time import sleep

import pandas as pd
import pytz
import vectorbt as vbt
import twstock

codes = twstock.codes

def clean_csv(csv_path):
    """ Clean the CSV file by removing rows with incorrect number of columns. """
    with open(csv_path, 'r') as file:
        lines = file.readlines()

    # Count the number of columns in the header
    num_columns = len(lines[0].split(','))

    with open(csv_path, 'w') as file:
        for line in lines:
            if len(line.split(',')) == num_columns:
                file.write(line)

def get_data_since_last_record(stock_num, base_path='./data/'):
    csv_path = f'{base_path}{stock_num}.csv'
    tz_taipei = pytz.timezone('Asia/Taipei')
    today = datetime.now(tz_taipei).replace(hour=0, minute=0, second=0, microsecond=0)  # Reset to start of day

    if os.path.exists(csv_path):
        try:
            clean_csv(csv_path)  # Clean the CSV file before reading
            data = pd.read_csv(csv_path, header=0)
        except pd.errors.ParserError as e:
            print(f"Parser error when reading CSV: {e}")
            data = pd.DataFrame()  # Fallback to an empty DataFrame

        if not data.empty:
            try:
                last_record_date = pd.to_datetime(data['Datetime'].iloc[-1]).tz_convert('Asia/Taipei')
                start_date = last_record_date + timedelta(minutes=5)
            except Exception as e:
                print(f"Error parsing last record date: {e}")
                start_date = today - timedelta(days=59)
        else:
            start_date = today - timedelta(days=59)
    else:
        start_date = today - timedelta(days=59)

    end_date = today + timedelta(hours=14)
    yf_data = vbt.YFData.download(
        f"{stock_num}.TW",
        start=start_date.strftime('%Y-%m-%d %H:%M:%S'),
        end=end_date.strftime('%Y-%m-%d %H:%M:%S'),
        interval='5m',
        missing_index='drop'
    )

    new_data = yf_data.get()

    if os.path.exists(csv_path):
        new_data.to_csv(csv_path, mode='a', header=False)
    else:
        new_data.to_csv(csv_path)
    sleep(1) # Sleep for 5 seconds to avoid rate limiting
    return new_data

for k, v in codes.items():
    if v.market == '上市' and (v.type == '股票' or v.type == 'ETF'):
        new_data = get_data_since_last_record(k)
        print(f"Updated data for {k}")
