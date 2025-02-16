import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
data = pd.read_csv("/Users/qyu/Downloads/ticker_data_0930.csv", index_col=0)
data = data.loc[:, ['event_time', 'sym', 'side', 'px', 'qty', 'lvl']]
print()

# please do some spread analysis here
# use the lvl = 1.0 data to calculate the spread and plot it
df = data[data['lvl'] == 1.0]
# remove the data where for each event_time, there are is no bid or ask
df = df.groupby('event_time').filter(lambda x: len(x) == 2)
# calculate the spread
df['spread'] = df.groupby('event_time')['px'].diff()
# remove the data where the spread is more than 5 standard deviation, store the outline data in df_outlier
df_outlier = df[np.abs(df['spread'] - df['spread'].mean()) > (5 * df['spread'].std())]
df = df[np.abs(df['spread'] - df['spread'].mean()) <= (5 * df['spread'].std())]
# describe the outline data
print(df_outlier)
print(df_outlier.describe())

# plot the spread
df.dropna(inplace=True)
df['spread'].plot()
plt.show()

# please do some volume analysis here
# calculate the volume for each event_time / snapshot
df['volume'] = df.groupby('event_time')['qty'].sum()
# plot the volume
df['volume'].plot()
plt.show()


# create some momentum indicators here
# get the spread from df
# calculate the spread
df['spread'] = df.groupby('event_time')['px'].diff()
# remove the data where the spread is more than 5 standard deviation, store the outline data in df_outlier
df_outlier = df[np.abs(df['spread'] - df['spread'].mean()) > (5 * df['spread'].std())]
df = df[np.abs(df['spread'] - df['spread'].mean()) <= (5 * df['spread'].std())]
# merge df (with only event_time and spread columns) and data
data = pd.merge(data, df[['event_time', 'spread']], on='event_time', how='left')
# if the spread is decreasing for 3 consecutive snapshots and the volume is increasing for 3 consecutive snapshots, then we say there is a trade flag
# construct a new column 'trade_flag' to store the trade flag
data['trade_flag'] = 0
# if spread is decreasing for 3 consecutive snapshots, then we say there is a trade flag
data['trade_flag'] = np.where((data['spread'].shift(1) > data['spread']) & (data['spread'].shift(2) > data['spread'].shift(1)) & (data['spread'].shift(3) > data['spread'].shift(2)), 1, data['trade_flag'])
# if the volume is increasing for 3 consecutive snapshots, then we say there is a trade flag
data['trade_flag'] = np.where((data['qty'].shift(1) < data['qty']) & (data['qty'].shift(2) < data['qty'].shift(1)) & (data['qty'].shift(3) < data['qty'].shift(2)), 1, data['trade_flag'])
# calculate a volumm weighted averge price and if the volumm weighted averge price is higher than the lvl 1,0 mid (the average of bid and ask), then we say there is a buy flag
data['vwap'] = data['px'] * data['qty']
data['vwap'] = data.groupby('event_time')['vwap'].transform('sum') / data.groupby('event_time')['qty'].transform('sum')
data['action'] = np.where(data['vwap'] > (data['bid'] + data['ask']) / 2, 1, data['trade_flag'])
# if the volumm weighted averge price is lower than the lvl 1,0 mid (the average of bid and ask), then we say there is a sell flag
data['action'] = np.where(data['vwap'] < (data['bid'] + data['ask']) / 2, -1, data['action'])
# else we say there is no trade flag
data['action'] = np.where(data['vwap'] == (data['bid'] + data['ask']) / 2, 0, data['action'])
# get final trade_flag using the action column
data['trade_flag'] *= data['action']


# write a backtest










