# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 17:01:29 2021

@author: Casey Macaluso
"""
### Imports
from pandas_datareader import data, wb
import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
import plotly
import plotly.io as pio
import cufflinks as cf
cf.go_offline()
pio.renderers.default = 'browser'

date_start = datetime(2006,1,1)
date_end = datetime(2016,1,1)

#### Finance DFs
BAC = data.DataReader("BAC",'stooq',date_start,date_end)
C = data.DataReader("C", 'stooq', date_start,date_end)
GS = data.DataReader("GS", 'stooq', date_start,date_end)
JPM = data.DataReader("JPM", 'stooq', date_start,date_end)
MS = data.DataReader("MS", 'stooq', date_start,date_end)
WFC = data.DataReader("WFC", 'stooq', date_start,date_end)

tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']

bank_stocks = pd.concat([BAC,C,GS,JPM,MS,WFC],keys=tickers,axis=1)
bank_stocks.columns.names = ['Ticker', 'Stock Info']

bank_stocks.head()
bank_stocks.info()

### Max close price for each bank's stock?
# bank_stocks.apply(lambda a: bank_stocks.xs(a,axis=1)['Close'].max())

for tick in tickers:
    print(bank_stocks.xs(tick,axis=1)['Close'].max())
    
### Returns
returns = pd.DataFrame()

for tick in tickers:
    returns[tick+' Return'] = bank_stocks.xs(tick,axis=1)['Close'].pct_change()  
# Check
returns.head(10)

### Pairplot
sns.pairplot(returns)
# C has a slightly higher return than most?

### Worst return days?
for col in returns.columns:
    print(col+'   '+str(returns[returns[col] == returns[col].min()].index[0]))
    
# 4 places had worst returns on same day (Jan 20, 2009 --> President Obama took office)

### Best Return days?
for col in returns.columns:
    print(col+'   '+str(returns[returns[col] == returns[col].max()].index[0]))
    
### Std Dev?
for col in returns.columns:
    print(col+'   '+str(returns[col].std()))

# C has highest std, may be riskiest investment (high risk, high reward)

### Distplot of returns
copy = returns.copy()
copy.reset_index(inplace=True)

ms_2015 = copy[pd.DatetimeIndex(copy['Date']).year == 2015]['MS Return']
sns.distplot(ms_2015,bins=50)

c_2008 = copy[pd.DatetimeIndex(copy['Date']).year == 2008]['C Return']
sns.distplot(c_2008,bins=50)



### More visualizations

### Close price for each bank
sns.lineplot(data=bank_stocks.xs('Close',level=1,axis=1))

### Rolling Avg vs. Closing price for BAC in 2008
bac_2008 = bank_stocks[pd.DatetimeIndex(bank_stocks.index).year == 2008].xs('BAC',axis=1)
bac_2008.head()
bac_2008['30 Avg Roll'] = bac_2008['Close'].rolling(window=30).mean()
sns.lineplot(data=bac_2008[['Close', '30 Avg Roll']]).set_title('Bank Of America (2008)')

### heatmap for closing prices
close = bank_stocks.xs('Close',level=1,axis=1)
sns.heatmap(close.corr(), annot=True, linecolor='black')

### clustermapping
sns.clustermap(close.corr())

### Cufflinks Library
# 2015 BAC candle plot
bac_2015_16 = bank_stocks.loc['2015-01-01':'2016-01-01'].xs('BAC',axis=1)
bac_2015_16.iplot(kind='candle')

# SMA plot for MS in 2015
ms_2015 = bank_stocks[pd.DatetimeIndex(bank_stocks.index).year == 2015].xs('MS',axis=1).drop('Volume',1)
ms_2015.ta_plot(study='sma')

# 2015 Bollinger Band Plot for BAC
bac_2015_16.drop('Volume',1).ta_plot(study='boll')
