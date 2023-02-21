# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 17:19:05 2021

@author: Casey Macaluso
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
df3 = pd.read_csv('df3')

df3.plot.scatter(x='a',y='b',c='red',s=50, figsize=(12,3))

df3['a'].plot.hist()

plt.style.use('ggplot')
df3['a'].plot.hist(bins=30)

df3[['a','b']].plot.box()
df3['d'].plot.density(linestyle='--',linewidth=3)
df3.iloc[0:30].plot.area(alpha=0.4).legend(loc='center left', bbox_to_anchor=(1,0.5))
