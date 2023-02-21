# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 15:17:43 2021

@author: Casey Macaluso
"""

import chart_studio.plotly as py
import pandas as pd
import numpy as np
from plotly import __version__
import cufflinks as cf
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
import plotly.io as pio

# Connects to interactive JS library
init_notebook_mode(connected=True)

cf.go_offline()

# **** THIS ALLOWS THE iplot() COMMAND TO GENERATE INTERACTIVE PLOTS IN BROWSER
# THESE PLOTS DO NOT WORK IN SPYDER BY DEFAULT
pio.renderers.default = 'browser'

df = pd.DataFrame(np.random.randn(100,4), columns='a b c d'.split())
df.head()

df2 = pd.DataFrame({'Category': ['A','B','C'],'Values': [32,43,50]})
df2

# NOTE: iplot does not appear to function in Spyder. Mainly a Jupyter Notebook feature, it seems
df.iplot() 

df.plot(kind='scatter',x='a',y='b')
df.iplot(kind='scatter',x='a',y='b',mode='markers')

df2.plot(kind='bar',x='Category',y='Values')
df2.iplot(kind='bar',x='Category',y='Values')

df.sum().iplot(kind='bar')

df.iplot(kind='box')

df3 = pd.DataFrame({'x':[1,2,3,4,5], 'y':[10,20,30,20,10], 'z':[5,4,3,2,1]})
df3.iplot(kind='surface', colorscale='rdylbu')

df['a'].iplot(kind='hist',bins=50)

# Can toggle specific columns with this
df.iplot(kind='hist')

# useful for stock/financial data
df[['a','b']].iplot(kind='spread')

df.iplot(kind='bubble',x='a',y='b',size='c')

df.scatter_matrix()
