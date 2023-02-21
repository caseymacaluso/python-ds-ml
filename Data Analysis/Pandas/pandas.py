# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 15:22:50 2021

@author: Casey Macaluso
"""

import numpy as np
import pandas as pd
import os

########################
######## Series ########
########################

labels = ['a', 'b', 'c']
data = [10,20,30]
arr = np.array(data)
d = {'a':10, 'b':20, 'c':30}

pd.Series(data=data)
pd.Series(data=data, index=labels)
pd.Series(data,labels)
pd.Series(arr,labels)
pd.Series(d)
pd.Series(labels)

ser1=pd.Series([1,2,3,4],['USA', 'USSR', 'Germany', 'Japan'])
ser1['USA']

ser2=pd.Series([1,2,5,4], ['USA', 'Italy', 'Germany', 'Japan'])
ser1 + ser2

############################
######## Dataframes ########
############################

from numpy.random import randn
np.random.seed(101)
df = pd.DataFrame(randn(5,4),['a','b','c','d','e'],['w', 'x', 'y', 'z'])
df['w']
df[['w','z']]
df['w']['a']
df['new'] = df['w'] + df['y']
df.drop('new', axis=1, inplace=True)
df.drop('e')
df.shape # (5,4) --> 5 is # of rows, in the 0 index. 4 is the # of columns, in the 1 index (USEFUL INFO FOR axis argument)
df.loc['a']
df.iloc[0]
df.loc['b','y']
df.loc[['a','b'],['w','y']]

booldf = df > 0
df[booldf]
df['w'] > 0
df[df['w'] > 0]['x']
df[df['z'] < 0]
df[(df['w'] > 0) & (df['y'] > 1)]
df[(df['w'] > 0) | (df['y'] > 1)]
df.reset_index() # not in place, must specify
newind = 'CA NY WY OR CO'.split()
df['states'] = newind
df.set_index('states') # not in place, must specify

out = ['g1', 'g1', 'g1', 'g2', 'g2', 'g2']
inside = [1,2,3,1,2,3]
heir_ind = list(zip(out, inside))
heir_ind = pd.MultiIndex.from_tuples(heir_ind)
df = pd.DataFrame(randn(6,2), heir_ind,['a','b'])
df.loc['g1'].iloc[2]
df.index.names = ['Groups', 'Num']
df.loc['g2'].loc[1]['a']
df.xs(1,level='Num')

##############################
######## Missing Data ########
##############################

d = {'a':[1,2,np.nan], 'b':[5,np.nan,np.nan], 'c':[1,2,3]}
df = pd.DataFrame(d)
df.dropna()
df.dropna(thresh=2)
df['a'].fillna(value=df['a'].mean())

##########################
######## Grouping ########
##########################

data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'], 'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'], 'Sales':[200,120,340,124,243,350]}
df = pd.DataFrame(data)
df_comp = df.groupby('Company')
df_comp.mean()
df_comp.std()
df_comp.sum().loc['FB']
df_comp.count()
df_comp.max()
df_comp.min()
df_comp.describe()

###########################################
######## Merge, Join & Concatenate ########
###########################################

df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'], 'B': ['B0', 'B1', 'B2', 'B3'], 'C': ['C0', 'C1', 'C2', 'C3'], 'D': ['D0', 'D1', 'D2', 'D3']}, index=[0, 1, 2, 3])
df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'], 'B': ['B4', 'B5', 'B6', 'B7'], 'C': ['C4', 'C5', 'C6', 'C7'], 'D': ['D4', 'D5', 'D6', 'D7']}, index=[4, 5, 6, 7]) 
df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],  'B': ['B8', 'B9', 'B10', 'B11'], 'C': ['C8', 'C9', 'C10', 'C11'], 'D': ['D8', 'D9', 'D10', 'D11']}, index=[8, 9, 10, 11])
pd.concat([df1,df2,df3])

left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'], 'A': ['A0', 'A1', 'A2', 'A3'], 'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'], 'C': ['C0', 'C1', 'C2', 'C3'], 'D': ['D0', 'D1', 'D2', 'D3']}) 
pd.merge(left,right,how='inner',on='key')

left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'], 'key2': ['K0', 'K1', 'K0', 'K1'], 'A': ['A0', 'A1', 'A2', 'A3'], 'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'], 'key2': ['K0', 'K0', 'K0', 'K0'], 'C': ['C0', 'C1', 'C2', 'C3'], 'D': ['D0', 'D1', 'D2', 'D3']})
pd.merge(left, right, on=['key1', 'key2'])
pd.merge(left, right, how='outer', on=['key1', 'key2'])
pd.merge(left, right, how='right', on=['key1', 'key2'])
pd.merge(left, right, how='left', on=['key1', 'key2'])

left = pd.DataFrame({'A': ['A0', 'A1', 'A2'], 'B': ['B0', 'B1', 'B2']}, index=['K0', 'K1', 'K2']) 
right = pd.DataFrame({'C': ['C0', 'C2', 'C3'], 'D': ['D0', 'D2', 'D3']}, index=['K0', 'K2', 'K3'])
left.join(right)
left.join(right, how='outer')

############################
######## Operations ########
############################

df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})
df['col2'].unique()
df['col2'].nunique()
df['col2'].value_counts()
df[df['col1'] > 2]

def t2(x):
    return x*2

df['col1'].apply(t2)
df['col3'].apply(len)
df['col2'].apply(lambda x: x*2)
df.columns
df.index
df.sort_values(by='col2')
df.isnull()

data = {'A':['foo','foo','foo','bar','bar','bar'], 'B':['one','one','two','two','one','one'], 'C':['x','y','x','y','x','y'], 'D':[1,3,2,5,4,1]}
df = pd.DataFrame(data)
df.pivot_table(values='D', index=['A','B'], columns=['C'])

###################################
######## Data Input/Output ########
###################################

os.chdir('Data Analysis/Pandas')
pd.read_csv("example")
df = pd.read_csv("example")
df.to_csv('my_output', index=False)
pd.read_csv('my_output')

pd.read_excel('Excel_Sample.xlsx',sheet_name='Sheet1')
df.to_excel('Excel_Sample2.xlsx',sheet_name='Sheet1')

data=pd.read_html('https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list/')
data[0].head()

from sqlalchemy import create_engine
engine = create_engine('sqlite:///:memory:')
df.to_sql('data', engine)
sql_df = pd.read_sql('data',con=engine)
