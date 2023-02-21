# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 13:59:29 2021

@author: Casey Macaluso
"""

import matplotlib.pyplot as plt
import numpy as np
import os


# Data
x = np.arange(0,100)
y = x*2
z = x**2

# Exercise 1
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(x,y)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('title')

# Exercise 2
fig = plt.figure()
ax1 = fig.add_axes([0,0,1,1])
ax2 = fig.add_axes([0.2,0.5,0.2,0.2])
ax1.plot(x,y)
ax1.set_xlabel('x')
ax2.set_ylabel('y')
ax2.plot(x,y)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
plt.tight_layout()

# Exercise 3
fig = plt.figure()
ax1 = fig.add_axes([0,0,1,1])
ax2 = fig.add_axes([0.2,0.5,0.4,0.4])
ax1.plot(x,z)
ax1.set_xlabel('x')
ax2.set_ylabel('z')
ax2.plot(x,y)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('zoom')
ax2.set_xlim([20,22])
ax2.set_ylim([30,50])

# Exercise 4
fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(7,2))
axes[0].plot(x,y,lw=2,ls='--',color='blue')
axes[1].plot(x,z,lw=3,color='red')
plt.tight_layout()
