# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 16:52:27 2021

@author: Casey Macaluso
"""

# 1
import numpy as np

# 2
np.zeros(10)

# 3
np.ones(10)

# 4
np.ones(10)*5

# 5
np.arange(10,51)

# 6
np.arange(10,51,2)

# 7
np.array([[0,1,2],[3,4,5],[6,7,8]])

# 8
np.eye(3)

# 9
np.random.rand(1)

# 10
np.random.randn(25)

# 11
arr = np.arange(1,101).reshape(10,10) / 100
# OR
np.linspace(0.01,1,100).reshape(10,10)

# 12
np.linspace(0,1,20)

# 13
mat = np.arange(1,26).reshape(5,5)
mat
# 14
mat[2:,1:]

# 15
mat[3,4]

# 16
mat[:3,1:2]

# 17
mat[4,:]

# 18
mat[3:,:]

# 19
mat.sum()

# 20
mat.std()

# 21
mat.sum(axis=0) #0=col, 1=row
