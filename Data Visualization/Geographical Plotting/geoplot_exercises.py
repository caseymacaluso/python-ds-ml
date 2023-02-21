# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 17:39:24 2021

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

# World Power Consumption
power = pd.read_csv('2014_World_Power_Consumption')
power.head()

powerdata = dict(type = 'choropleth',
                 locations = power['Country'],
                 locationmode='country names', # Include here, as we don't have country codes
                 z = power['Power Consumption KWH'],
                 text = power['Text'],
                 colorbar = {'title': 'Power Consumption in KWH'})

powerlay = dict(title = '2014 Global Power Consumption',
              geo = dict(showframe=False,
                         projection = {'type': 'mercator'}))

choromap = go.Figure([powerdata], powerlay)
iplot(choromap,validate=False)


# 2012 Election
elect = pd.read_csv('2012_Election_Data')
elect.head()

electdata = dict(type = 'choropleth',
                 colorscale='Viridis',
                 locations = elect['State Abv'],
                 locationmode='USA-states',
                 reversescale=True, # flips shading so larger values are darker
                 z = elect['Voting-Age Population (VAP)'],
                 text = elect['State'],
                 colorbar = {'title': 'Voting Age Population in the U.S.'})

electlay = dict(title = '2012 Voting Age Population', geo={'scope':'usa'})

choromap2 = go.Figure([electdata], electlay)
iplot(choromap2, validate=False)
