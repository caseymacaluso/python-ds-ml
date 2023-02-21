# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 16:38:07 2021

@author: Casey Macaluso
"""

import pandas as pd
import os
import chart_studio.plotly as py
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
import plotly.io as pio
import plotly.graph_objs as go
pio.renderers.default = 'browser'

#################################
######## Choropleth Maps ########
#################################

data = dict(type = 'choropleth', 
            locations = ['AZ','CA','NY'], 
            locationmode= 'USA-states', 
            colorscale = 'Portland', 
            text=['Arizona','California', 'New York'],
            z = [1.0,2.0,3.0],
            colorbar = {'title': 'Colorbar title goes here'})

layout = dict(geo={'scope':'usa'})
choromap = go.Figure(data=[data], layout=layout)
iplot(choromap)

df = pd.read_csv('2011_US_AGRI_Exports')
df.head()

data = dict(type='choropleth', 
            colorscale = 'ylorbr', 
            locations = df['code'], 
            locationmode = 'USA-states', 
            z = df['total exports'],
            text = df['text'],
            marker = dict(line = dict(color='rgb(12,12,12)',width=2)),
            colorbar = {'title': 'Millions USD'}
            )

layout = dict(title = '2011 US Agricultural Exports by State',
              geo = dict(scope='usa', showlakes=True, lakecolor='rgb(85,173,240)'))

choromap2 = go.Figure([data], layout)
iplot(choromap2)


df = pd.read_csv('2014_World_GDP')
df.head()

data = dict(type='choropleth',
            locations = df['CODE'],
            z = df['GDP (BILLIONS)'],
            text = df['COUNTRY'],
            colorbar = {'title': 'GDP in Billions USD'})

layout = dict(title = '2014 Global GDP',
              geo = dict(showframe=False,
                         projection = {'type': 'mercator'})) # refer to docs for different projection types to mess around with

choromap3 = go.Figure([data], layout)
iplot(choromap3)
