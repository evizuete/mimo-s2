from datetime import datetime

import pandas as pd

from mimo.data_managers.databases import Database

def get_rates_from_file(file):
    cols = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'spread']
    df = pd.read_csv(file, header=0, names=cols)
    df['time'] = pd.to_datetime(df.datetime)
    df = df.drop(['datetime', 'spread'], axis=1)

    return df

file = './data.csv'
#file = "./data_last_week.csv"
df = get_rates_from_file(file)

db = Database()
#db.save_massive_data(df, 'rates', chunk_size=10000)
db.save_massive_data(df, 'historical_rates', chunk_size=10000)

print('')