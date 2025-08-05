import pandas as pd
import os 
import calendar
import math 
from datetime import datetime

input_dir = r'raw_csvs'
output_dir = r'cleaned_csvs'
output_file = os.path.join(output_dir, 'CondensateTemp.csv')

os.makedirs(output_dir, exist_ok=True)


dfs = []
meter_names = []

for file in os.listdir(input_dir):
    if file.endswith('.csv'):
        inputFilePath = os.path.join(input_dir, file)
        try:
            df = pd.read_csv(inputFilePath, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(inputFilePath, encoding="latin1")
        meter = df['Sensor Name'].iloc[0][14:].strip()
        meter_names.append(meter)
        df = df[['Date Time', 'Temp (°C)']].copy()
        df = df.rename(columns={'Temp (°C)': meter})
        dfs.append(df)

from functools import reduce
merged = reduce(lambda left, right: pd.merge(left, right, on='Date Time', how='outer'), dfs)

merged['Date Time'] = pd.to_datetime(merged['Date Time'], dayfirst=True, errors='coerce')
merged = merged.sort_values('Date Time')
merged = merged.dropna(subset=['Date Time'])

merged['Year'] = merged['Date Time'].dt.year
merged['Month'] = merged['Date Time'].dt.month.map(lambda x: calendar.month_abbr[x])
merged['Day'] = merged['Date Time'].dt.strftime('%a')
merged['Date'] = merged['Date Time'].dt.date
merged['Time'] = merged['Date Time'].dt.time
merged['Week'] = merged['Date Time'].dt.dayofyear.apply(lambda x: math.ceil(x / 7))

cols = ['Year', 'Month', 'Day', 'Date', 'Time', 'Week'] + meter_names
merged = merged[cols]

if os.path.exists(output_file):
    existing_df = pd.read_csv(output_file)
    combined = pd.concat([existing_df, merged], ignore_index=True)
    combined = combined.drop_duplicates(subset=['Year', 'Month', 'Day', 'Date', 'Time'] + meter_names)
    combined.to_csv(output_file, index=False)
    print(f"Appended new data to {output_file}")
else:
    merged.to_csv(output_file, index=False)
    print(f"Saved cleaned data to {output_file}")
