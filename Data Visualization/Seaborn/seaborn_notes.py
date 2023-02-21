# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 14:20:21 2021

@author: Casey Macaluso
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

####################################
######## Distribution Plots ########
####################################

tips = sns.load_dataset('tips')
tips.head()

sns.distplot(tips['total_bill'], kde=False, bins=30)
sns.jointplot(x='total_bill',y='tip',data=tips)
sns.pairplot(tips, hue='sex', palette='coolwarm') #plots for all numerical information
sns.kdeplot(tips['total_bill'])
sns.rugplot(tips['total_bill'])

###################################
######## Categorical Plots ########
###################################

sns.barplot(x='sex',y='total_bill',data=tips, estimator=np.std)
sns.countplot(x='sex',data=tips)
sns.boxplot(x='day',y='total_bill',data=tips) # 'hue' can add another layer of analysis by adding another column
sns.violinplot(x='day',y='total_bill', data=tips, hue='sex', split=True)
sns.stripplot(x='day',y='total_bill',data=tips, jitter=True, hue='sex',split=True)
sns.swarmplot(x='day',y='total_bill',data=tips) #similar to violin plots, just shows the individual points (not recommended for large datasets)

sns.violinplot(x='day',y='total_bill', data=tips)
sns.swarmplot(x='day',y='total_bill',data=tips, color='black')

# Factorplot --> renamed to 'catplot'
sns.catplot(x='day',y='total_bill',data=tips,kind='bar')

##############################
######## Matrix Plots ########
##############################

flights = sns.load_dataset('flights')
flights.head()
tips.corr()
sns.heatmap(tips.corr(),annot=True,cmap='coolwarm')
fp = flights.pivot_table(index='month',columns='year',values='passengers')
sns.heatmap(fp,cmap='magma',linecolor='white',linewidths=1)
sns.clustermap(fp, standard_scale=1) # Hierarchical clustering, can see which months/years are most similar
# standard_scale standardizes the data used within a clustermap.

#######################
######## Grids ########
#######################

iris = sns.load_dataset('iris')
iris.head()
g = sns.PairGrid(iris)
g.map(plt.scatter)
# OR
g.map_diag(sns.distplot)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot) # More control this way

g = sns.FacetGrid(tips, col='time', row='smoker')
g.map(sns.distplot,'total_bill')

g = sns.FacetGrid(tips, col='time', row='smoker')
g.map(plt.scatter,'total_bill','tip')


##################################
######## Regression Plots ########
##################################

sns.lmplot(x='total_bill',y='tip',data=tips,hue='sex',markers=['o','v'])
sns.lmplot(x='total_bill',y='tip',data=tips,hue='sex',markers=['o','v'], scatter_kws={'s':50})

sns.lmplot(x='total_bill',y='tip',data=tips,col='sex',row='time')
sns.lmplot(x='total_bill',y='tip',data=tips,col='sex',row='time',aspect=0.6,size=8)


###############################
######## Style & Color ########
###############################

sns.set_style('ticks')
plt.figure(figsize=(12,3)) # Can use matplotlib figure size adjustments with seaborn plots
sns.set_context('notebook')
sns.countplot(x='sex', data=tips)
sns.despine()

sns.lmplot(x='total_bill',y='tip',data=tips,hue='sex',palette='seismic') # refer to docs for palette strings

