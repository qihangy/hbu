import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
data = pd.read_csv("/Users/qyu/Desktop/hbu_analytics/BTC_minute_bar.csv", index_col=0)
# do some regression analysis here
# calculate the return
data['return'] = data['close'].pct_change()
# Shift returns to align with indicators for prediction purposes
data['lagged_return'] = data['return'].shift(-1)
# drop the nan data
data.dropna(inplace=True)
# regression the return with volume
# import the statsmodels.api
import statsmodels.api as sm
# define the independent variable
X = data['volume']
# define the dependent variable
y = data['return']
# add a constant to the independent variable
X = sm.add_constant(X)
# fit the regression model
model = sm.OLS(y, X).fit()
# print the summary of the regression model
print(model.summary())

# calculate the RSI indicator
# idea of RSI indicator: if the price goes up, then the RSI goes up, if the price goes down, then the RSI goes down
# the full name of RSI is relative strength index, it is a momentum indicator, it is used to measure the magnitude of
# recent price changes to evaluate overbought or oversold conditions in the price of a stock or other asset
# define the period
period = 14
# calculate the change of price
data['price_change'] = data['close'].diff()
# remove the nan data
data.dropna(inplace=True)
# define the up and down
data['up'] = np.where(data['price_change'] > 0, data['price_change'], 0)
data['down'] = np.where(data['price_change'] < 0, data['price_change'], 0)
# calculate the average gain and average loss
data['avg_gain'] = data['up'].rolling(period).mean()
data['avg_loss'] = data['down'].abs().rolling(period).mean()
# calculate the relative strength
data['rs'] = data['avg_gain'] / data['avg_loss']
# calculate the RSI
data['rsi'] = 100 - (100 / (1 + data['rs']))
# plot the RSI
data['rsi'].dropna().plot()

# do the analysis of the RSI using the regression method
# define the independent variable
X = data['rsi']
# define the dependent variable
y = data['lagged_return']
# add a constant to the independent variable
X = sm.add_constant(X)
# fit the regression model
model = sm.OLS(y, X).fit()
# print the summary of the regression model
print(model.summary())

# volume vwap indicator
# calculate the volume vwap
data['volume_vwap'] = data['volume'] * data['vwap']
# calculate the 30 minute rolling volume vwap
data['volume_vwap_30'] = data['volume_vwap'].rolling(30).sum()
# calculate the 30 minute rolling volume
data['volume_30'] = data['volume'].rolling(30).sum()
# calculate the 30 minute rolling volume vwap
data['volume_vwap_30'] = data['volume_vwap_30'] / data['volume_30']
# plot the volume vwap
data['volume_vwap_30'].plot()
plt.show()

# do the analysis of the volume vwap using the regression method

# define the independent variable
X = data['volume_vwap_30']
# define the dependent variable
y = data['lagged_return']
# add a constant to the independent variable
X = sm.add_constant(X)
# fit the regression model
model = sm.OLS(y, X).fit()
# print the summary of the regression model
print(model.summary())

# MACD indicator
# calculate the 12 period exponential moving average
data['ema_12'] = data['close'].ewm(span=12).mean()
# calculate the 26 period exponential moving average
data['ema_26'] = data['close'].ewm(span=26).mean()
# calculate the MACD
data['macd'] = data['ema_12'] - data['ema_26']
# calculate the 9 period exponential moving average of MACD
data['ema_macd'] = data['macd'].ewm(span=9).mean()
# plot the MACD
data['macd'].plot()
plt.show()

# do the analysis of the MACD using the regression method
# define the independent variable
X = data['macd']
# define the dependent variable
y = data['lagged_return']
# add a constant to the independent variable
X = sm.add_constant(X)
# fit the regression model
model = sm.OLS(y, X).fit()
# print the summary of the regression model
print(model.summary())



#
# # please do liquidity analysis here
# # the liquidity is defined as the ratio of the 30 minute rolling standard deviation of price return (close - open) / open to the volume
# # calculate the liquidity
# data['liquidity'] = data['close'].rolling(30).std() / data['volume']
# # drop the data where the liquidity is nan
# data.dropna(inplace=True)
# # plot the liquidity
# data['liquidity'].plot()
# plt.show()
#
# # calculate the rolling z score of liquidity for each 60 minutes
# data['liquidity_z_score'] = data['liquidity'].rolling(60).apply(lambda x: (x[-1] - x.mean()) / x.std())
# # plot the rolling z score of liquidity
# data['liquidity_z_score'].plot()
# plt.show()
#
# # apply the regine change method to find the regime change points
# # smooth the rolling z score of liquidity by using the rolling median
# data['liquidity_z_score'] = data['liquidity_z_score'].rolling(5).median()
# # model a regime swtching model based on the rolling z score of liquidity
# # if the rolling z score of liquidity is lower than -1.5, then we say there is a regime change point
# data['regime_change'] = np.where(data['liquidity_z_score'] < -1.5, 1, 0)
# # if the rolling z score of liquidity is between -1.5 and 1.5, then we say there is no regime change point
# data['regime_change'] = np.where((data['liquidity_z_score'] >= -1.5) & (data['liquidity_z_score'] <= 1.5), 0, data['regime_change'])
# # if the rolling z score of liquidity is higher than 1.5, then we say there is a regime change point
# data['regime_change'] = np.where(data['liquidity_z_score'] > 1.5, 1, data['regime_change'])
#
datetime_str = "2023-12-13 03:00:02.811892"

# Convert to datetime object
datetime_object = pd.to_datetime(datetime_str)