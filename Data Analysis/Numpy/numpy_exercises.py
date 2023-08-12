# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 16:52:27 2021

@author: Casey Macaluso
"""

# 1
import numpy as np

# 2: Array of ten 0's
np.zeros(10)

# 3: Array of ten 1's
np.ones(10)

# 4: Array of ten 5's
np.ones(10)*5

# 5: Array of all #s from 10-50
np.arange(10,51)

# 6: Array of all even #'s from 10-50
np.arange(10,51,2)

# 7: 3x3 Array from 0-8
np.array([[0,1,2],[3,4,5],[6,7,8]])

# 8: 3x3 Array with 1's on the diagonal
np.eye(3)

# 9: Random # between 0 and 1
np.random.rand(1)

# 10: 25 random numbers from normal distribution
np.random.randn(25)

# 11: 10x10 array with values from 0-1, with lowest at 0.01, highest at 1
arr = np.arange(1,101).reshape(10,10) / 100
# OR
np.linspace(0.01,1,100).reshape(10,10)

# 12: 20 linearly spaced points from 0-1
np.linspace(0,1,20)

# 13
mat = np.arange(1,26).reshape(5,5)
mat
# 14: Selects the third row onwards, and the second column onwards
mat[2:,1:]

# 15: Just the # 20
mat[3,4]

# 16: Everything up to the fourth row, first column
mat[:3,1:2]

# 17: Fifth row
mat[4,:]

# 18: Last two rows
mat[3:,:]

# 19: Sum all values in the matrix
mat.sum()

# 20: Standard deviation of all values matrix
mat.std()

# 21: Sum of all the columns in the matrix
mat.sum(axis=0) #0=col, 1=row
