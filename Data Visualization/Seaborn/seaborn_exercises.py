# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 16:55:43 2021

@author: Casey Macaluso
"""

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
titanic = sns.load_dataset('titanic')
titanic.head()

sns.jointplot(x='fare',y='age',data=titanic)

sns.distplot(titanic['fare'],kde=False,color='red',bins=30)

sns.boxplot(x='class',y='age',data=titanic)

sns.swarmplot(x='class',y='age',data=titanic)

sns.countplot(x='sex',data=titanic)

sns.heatmap(titanic.corr(),cmap='coolwarm')

g = sns.FacetGrid(titanic,col='sex')
g.map(sns.distplot,'age', kde=False)
