# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 14:59:45 2021

@author: Casey Macaluso
"""

# Imports
import pandas as pd
import numpy as np
# os.chdir('Capstone Projects')
import os
import seaborn as sns
import matplotlib.pyplot as plt
import chart_studio.plotly as py
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
import plotly.io as pio
import plotly.graph_objs as go
pio.renderers.default = 'browser'

df = pd.read_csv('911.csv')
df.info()
df.head()

### Top 5 zipcodes for 911 calls?
zipgrp = df.groupby('zip').sum()
zipgrp['e'].sort_values(ascending=False).head()
# OUT:
# zip
# 19401.0    6979
# 19464.0    6643
# 19403.0    4854
# 19446.0    4748
# 19406.0    3174

### Top 5 townships for 911 calls?
twpgrp = df.groupby('twp').sum()
twpgrp['e'].sort_values(ascending=False).head()
# OUT:
# twp
# LOWER MERION    8443
# ABINGTON        5977
# NORRISTOWN      5890
# UPPER MERION    5227
# CHELTENHAM      4575

### Unique title codes?
df['title'].nunique()
# OUT:
# 110

### New 'Reason' column based on 'Reasons/Department' from title column
df['Reasons'] = df['title'].apply(lambda s: s.split(':')[0])
# Check
df[['title', 'Reasons']].head(10)
# OUT:
# 0       EMS: BACK PAINS/INJURY      EMS
# 1      EMS: DIABETIC EMERGENCY      EMS
# 2          Fire: GAS-ODOR/LEAK     Fire
# 3       EMS: CARDIAC EMERGENCY      EMS
# 4               EMS: DIZZINESS      EMS
# 5             EMS: HEAD INJURY      EMS
# 6         EMS: NAUSEA/VOMITING      EMS
# 7   EMS: RESPIRATORY EMERGENCY      EMS
# 8        EMS: SYNCOPAL EPISODE      EMS
# 9  Traffic: VEHICLE ACCIDENT -  Traffic

### Most common reason for 911 call based off 'Reasons'?
df.groupby('Reasons').sum()['e'].sort_values(ascending=False)
# OUT:
# EMS        48877
# Traffic    35695
# Fire       14920

### Seaborn countplot by 'Reason'
sns.countplot(x='Reasons', data=df)

### Data type of timeStamp?
df['timeStamp'].dtypes

### Convert to datetime?
df['timeStamp'] = pd.to_datetime(df['timeStamp'])

### Create hour, month and day columns based off 'timeStamp'?
df['Hour'] = df['timeStamp'].apply(lambda t: t.hour)
df['Month'] = df['timeStamp'].apply(lambda t: t.month)
df['Day'] = df['timeStamp'].apply(lambda t: t.day_of_week)
# Check
df[['timeStamp', 'Hour', 'Month', 'Day']].head(15)

### Map dictionary to string names for day of week
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day'] = df['Day'].map(dmap)


### Seaborn countplot w/ DOW and 'reasons' as hue
sns.countplot(x='Day',data=df,hue='Reasons').legend(loc='center left', bbox_to_anchor=(1,0.5))

### Seaborn countplot w/ Month
sns.countplot(x='Month',data=df,hue='Reasons').legend(loc='center left', bbox_to_anchor=(1,0.5))
# There are three months missing here...why?

### Group month
byMonth = df.groupby('Month').count()
byMonth.head()

### Plot for # calls
byMonth.plot.line(y='e')

### lmplot for number of calls per month
sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())

### 'Date' column
df['Date'] = df['timeStamp'].apply(lambda t: t.date())
byDate = df.groupby('Date').count()
sns.lineplot(data=byDate, x='Date',y='e')

### Same as above, but different 'Reasons'
byDateTraffic = df[df['Reasons'] == 'Traffic'].groupby('Date').count()
sns.lineplot(data=byDateTraffic, x='Date',y='e').set_title('Traffic')

byDateFire = df[df['Reasons'] == 'Fire'].groupby('Date').count()
sns.lineplot(data=byDateFire, x='Date',y='e').set_title('Fire')

byDateEMS = df[df['Reasons'] == 'EMS'].groupby('Date').count()
sns.lineplot(data=byDateEMS, x='Date',y='e').set_title('EMS')

### Restructure df into pivot table
df_piv = df.groupby(['Hour','Day']).count()['e'].unstack(0)
### Heatmap
sns.heatmap(df_piv,cmap=sns.cm.rocket_r)
### Clustermap
sns.clustermap(df_piv)

### Pivot w/ different format
df_piv2 = df.groupby(['Month','Day']).count()['e'].unstack(0)
### Heatmap
sns.heatmap(df_piv2,cmap=sns.cm.rocket_r)
### Clustermap
sns.clustermap(df_piv2)
