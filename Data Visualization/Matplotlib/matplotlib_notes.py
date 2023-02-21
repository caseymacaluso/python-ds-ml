# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 13:07:43 2021

@author: Casey Macaluso
"""

import matplotlib.pyplot as plt
import numpy as np
import os

########################
######## Part 1 ########
########################
x = np.linspace(0,5,11)
y = x ** 2

# Functional
# Run all of these together to get labels on plot
plt.plot(x,y)
plt.show() # Plots appear in Plots pane by default, no need for this in Spyder, but other IDE's may need this.
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Title')

plt.subplot(1,2,1)
plt.plot(x,y,'r')

plt.subplot(1,2,2)
plt.plot(y,x,'b')


# Object Oriented
fig = plt.figure()
axes = fig.add_axes([0.1,0.1,0.8,0.8])
axes.plot(x,y)
axes.set_xlabel('X Label')
axes.set_ylabel('Y Label')
axes.set_title('Title')


fig = plt.figure()
axes1 = fig.add_axes([0.1,0.1,0.8,0.8])
axes2 = fig.add_axes([0.2,0.5,0.4,0.3])
axes1.plot(x,y)
axes1.set_title('Big One')
axes2.plot(y,x)
axes2.set_title('Smol One')

########################
######## Part 2 ########
########################

fig,axes = plt.subplots(nrows=1,ncols=2)
# for ax in axes:
#     ax.plot(x,y)
axes[0].plot(x,y)
axes[0].set_title('First')
axes[1].plot(y,x)
axes[1].set_title('Second')
plt.tight_layout()

# Figure Size, Aspect Ratio, DPI
fig = plt.figure(figsize=(8,2))
ax = fig.add_axes([0,0,1,1])
ax.plot(x,y)

fig,axes = plt.subplots(nrows=2,ncols=1,figsize=(8,2))
axes[0].plot(x,y)
axes[1].plot(y,x)
plt.tight_layout()

os.chdir('Data Visualization/Matplotlib')
fig.savefig('my_example_figure.png', dpi=300)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(x,x**2,label='x square')
ax.plot(x,x**3, label='x cube')
ax.legend(loc=0) # check docs for location codes


########################
######## Part 3 ########
########################

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(x,y,color='#ff8c00', lw='1', ls='--',marker='o', markersize=10, 
        markerfacecolor='purple', markeredgewidth=3, markeredgecolor='blue') # Check docs for different line styles, etc.
ax.set_xlim([0,1])
ax.set_ylim([0,2])

# Special Plots - will most likely use Seaborn for these
plt.scatter(x,y)




