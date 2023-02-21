# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 17:17:44 2021

@author: Casey Macaluso
"""

import os
import pandas as pd
import seaborn as sns
import numpy as np
os.chdir('../Pandas')

df1 = pd.read_csv('df1',index_col=0)
df2 = pd.read_csv('df2')
df1.head()
df2.head()

# Different ways to plot with pandas
df1['A'].hist()
df1['A'].plot(kind='hist')
df1['A'].plot.hist()

df2.plot.area(alpha=0.4)
df2.plot.bar()
df2.plot.bar(stacked=True)

df1.plot.line(y='B',figsize=(12,3),lw=1)

df1.plot.scatter(x='A',y='B')
df1.plot.scatter(x='A',y='B',c='C',cmap='coolwarm') # Can pass in custom color configs and maps
df1.plot.scatter(x='A',y='B',s=df1['C']*100)

df2.plot.box()
df = pd.DataFrame(np.random.randn(1000,2),columns=['a','b'])
df.plot.hexbin(x='a',y='b',gridsize=25)
df2['a'].plot.kde()
df2['a'].plot.density()
df2.plot.kde()
