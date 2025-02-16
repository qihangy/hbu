"""
Compute the Technical Indicators on the raw dataset:

Moving Averages: SMA_5, SMA_30, EMA_10, EMA_50
Momentum Indicators: ROC, RSI, Momentum, MACD
Volatility Indicators: BollingerB_Middle, BollingerB_Upper, BollingerB_Lower, Standard_Deviation, ATR
Volume Indicators: OBV, MFI
Support & Resistance: Pivot_Point, Resistance1, Support1, Resistance2, Support2

Drop NaN values & Calculate Lagged Return.

Perform the Multiple Regression using the above indicators to predict the Lagged_Return.

"""

# Importing the libraries
import pandas as pd
import numpy as np
pd.options.display.float_format = '{:.4f}'.format
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Importing the dataset
import pandas as pd
import numpy as np
pd.options.display.float_format = '{:.4f}'.format
import lovely_logger as log
import matplotlib.pyplot as plt


# Loading the raw dataset again
btc_data_raw = pd.read_csv('BTC_minute_bar.csv',index_col=0)

# Moving Averages
btc_data_raw['SMA_5'] = btc_data_raw['close'].rolling(window=5).mean()
btc_data_raw['SMA_30'] = btc_data_raw['close'].rolling(window=30).mean()
btc_data_raw['EMA_10'] = btc_data_raw['close'].ewm(span=10, adjust=False).mean()
btc_data_raw['EMA_50'] = btc_data_raw['close'].ewm(span=50, adjust=False).mean()
# create the cross over signal
btc_data_raw['SMA_5_30'] = np.where(btc_data_raw['SMA_5'] > btc_data_raw['SMA_30'], 1, 0)
btc_data_raw['EMA_10_50'] = np.where(btc_data_raw['EMA_10'] > btc_data_raw['EMA_50'], 1, 0)
# combine the cross over signal
btc_data_raw['SMA_5_30_EMA_10_50'] = np.where((btc_data_raw['SMA_5_30'] == 1) & (btc_data_raw['EMA_10_50'] == 1), 1, 0)

# Momentum Indicators
btc_data_raw['ROC'] = btc_data_raw['close'].pct_change(periods=1)
btc_data_raw['Momentum'] = btc_data_raw['close'] - btc_data_raw['close'].shift(4)
btc_data_raw['MACD'] = btc_data_raw['close'].ewm(span=12, adjust=False).mean() - btc_data_raw['close'].ewm(span=26, adjust=False).mean()
# RSI
delta = btc_data_raw['close'].diff()
gain = (delta.where(delta > 0, 0)).fillna(0)
loss = (-delta.where(delta < 0, 0)).fillna(0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
btc_data_raw['RSI'] = 100 - (100 / (1 + rs))
# Stochastic Oscillator
btc_data_raw['L14'] = btc_data_raw['low'].rolling(window=14).min()
btc_data_raw['H14'] = btc_data_raw['high'].rolling(window=14).max()
btc_data_raw['%K'] = 100 * ((btc_data_raw['close'] - btc_data_raw['L14']) / (btc_data_raw['H14'] - btc_data_raw['L14']))
btc_data_raw['%D'] = btc_data_raw['%K'].rolling(window=3).mean()
# Williams %R
btc_data_raw['%R'] = -100 * ((btc_data_raw['H14'] - btc_data_raw['close']) / (btc_data_raw['H14'] - btc_data_raw['L14']))
# create the momentum cross over signal
btc_data_raw['ROC_Threshold'] = np.where(btc_data_raw['ROC'] > 0, 1, 0)
btc_data_raw['Momentum_Threshold'] = np.where(btc_data_raw['Momentum'] > 0, 1, 0)
btc_data_raw['MACD_Threshold'] = np.where(btc_data_raw['MACD'] > 0, 1, 0)
btc_data_raw['RSI_Threshold'] = np.where(btc_data_raw['RSI'] > 50, 1, 0)
btc_data_raw['%K_Threshold'] = np.where(btc_data_raw['%K'] > 50, 1, 0)
btc_data_raw['%D_Threshold'] = np.where(btc_data_raw['%D'] > 50, 1, 0)
btc_data_raw['%R_Threshold'] = np.where(btc_data_raw['%R'] > 50, 1, 0)
# todo combine all the momentum cross over signal


# Volatility Indicators
btc_data_raw['BollingerB_Middle'] = btc_data_raw['close'].rolling(window=20).mean()
std_dev = btc_data_raw['close'].rolling(window=20).std()
btc_data_raw['BollingerB_Upper'] = btc_data_raw['BollingerB_Middle'] + (std_dev * 2)
btc_data_raw['BollingerB_Lower'] = btc_data_raw['BollingerB_Middle'] - (std_dev * 2)
btc_data_raw['Standard_Deviation'] = std_dev
btc_data_raw["ATR"] = (btc_data_raw["high"] - btc_data_raw["low"]).rolling(window=14).mean()

# Volume Indicators
btc_data_raw['OBV'] = (btc_data_raw['volume'] * (~btc_data_raw['close'].diff().le(0) * 2 - 1)).cumsum()
# MFI
typical_price = (btc_data_raw['high'] + btc_data_raw['low'] + btc_data_raw['close']) / 3
money_flow = typical_price * btc_data_raw['volume']
positive_money_flow = money_flow.where(btc_data_raw['close'].diff(1) > 0, 0).rolling(window=14).sum()
negative_money_flow = money_flow.where(btc_data_raw['close'].diff(1) < 0, 0).rolling(window=14).sum()
money_flow_ratio = positive_money_flow / negative_money_flow
btc_data_raw['MFI'] = 100 - (100 / (1 + money_flow_ratio))

# Support & Resistance
btc_data_raw['Pivot_Point'] = (btc_data_raw['high'] + btc_data_raw['low'] + btc_data_raw['close']) / 3
btc_data_raw['Resistance1'] = 2 * btc_data_raw['Pivot_Point'] - btc_data_raw['low']
btc_data_raw['Support1'] = 2 * btc_data_raw['Pivot_Point'] - btc_data_raw['high']
btc_data_raw['Resistance2'] = btc_data_raw['Pivot_Point'] + (btc_data_raw['high'] - btc_data_raw['low'])
btc_data_raw['Support2'] = btc_data_raw['Pivot_Point'] - (btc_data_raw['high'] - btc_data_raw['low'])

# Drop NaN values
btc_data_raw.dropna(inplace=True)

# Calculate the Lagged Return
btc_data_raw['Return'] = btc_data_raw['close'].pct_change()
btc_data_raw['Lagged_Return'] = btc_data_raw['Return'].shift(1)
btc_data_cleaned = btc_data_raw.dropna()

btc_data_cleaned.head()

# correlation matrix
# 1. Stationarity: Augmented Dickey-Fuller test for Lagged_Return
adf_result = adfuller(btc_data_cleaned['Lagged_Return'])
adf_pvalue = adf_result[1]

# 2. Multicollinearity: Correlation matrix for the technical indicators
technical_indicators = ['SMA_5', 'SMA_30', 'EMA_10', 'EMA_50', 'SMA_5_30', 'EMA_10_50', 'ROC', 'Momentum', 'MACD',
                        'RSI', '%K', '%D', '%R', 'BollingerB_Middle',
                        'BollingerB_Upper', 'BollingerB_Lower', 'Standard_Deviation', 'ATR', 'OBV', 'MFI', 'Pivot_Point', 'Resistance1', 'Support1', 'Resistance2', 'Support2']
correlation_matrix = btc_data_cleaned[technical_indicators].corr()

# Visualizing the correlation matrix
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Technical Indicators')
plt.show()
