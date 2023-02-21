# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 15:59:34 2021

@author: Casey Macaluso
"""
import numpy as np

########################
######## Arrays ########
########################
l = [1,2,3]
arr = np.array(l)
arr
mat = [[1,2,3], [4,5,6], [7,8,9]]
np.array(mat)

# Numpy Methods
np.arange(0,10,2)
np.zeros(5)
np.zeros((5,5)) # zeros
np.ones((3,3)) # ones
np.linspace(0,5,10) # evenly spaced numbers
np.eye(4) # Identity
np.random.rand(3,4) # Uniform dist
np.random.randn(3,4) # Standard normal dist
np.random.randint(1,100,5)

arr = np.arange(25)
ranarr = np.random.randint(0,50,10)
arr.reshape(5,5)
ranarr.max()
ranarr.min()
ranarr.argmax()
ranarr.argmin()
arr.shape
arr.dtype

##########################
######## Indexing ########
##########################

arr = np.arange(0,11)
arr[8]
arr[1:5]
arr[5:]
arr[0:5] = 100 # Casting
arr = np.arange(0,11)
arr_slice = arr[0:6] # Acts as view to arr
arr_slice[:] = 99
arr # Affects original array as well
arr2 = arr.copy()
arr2[:] = 100

# 2D Arrays
arr_2d = np.array([[5,10,15], [20, 25, 30], [35, 40, 45]])
arr_2d[0][0]
arr_2d[1,2]
arr_2d[:2,1:]

# Conditional Selection
arr = np.arange(0,11)
bool_arr = arr > 5
arr[bool_arr]
arr[arr<3]

############################
######## Operations ########
############################

arr + arr
arr * arr
2*arr
np.sqrt(arr)
np.exp(arr)
arr.max()
np.sin(arr)
np.log(arr)
