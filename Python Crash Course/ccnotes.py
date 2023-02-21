# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 18:22:03 2021

@author: Casey Macaluso
"""
#####################
####### Part 1 ######
#####################

# Strings
num = 12
name = 'Casey'
print('My number is {} and my name is {}.'.format(num, name))
print('My number is {one} and my name is {two}.'.format(one=num, two=name))

# String Slicing
name[0]
name[0:]
name[:3]
name[2:5]

# Lists
list1 = ['a', 'b', 'c']
list1.append('d')
list1[3]
nest = [1, 2, [3, 4, 5]]
nest[2][2]

#####################
###### Part 2 ######
#####################

# Dictionaries
d = {'k1': 'val1', 'k2': 'val2'}
d['k1']
d2 = {'k1': {'ik': [1,2,3]}}
d2['k1']['ik'][1]

# Tuples
t = (1,2,3,4) # IMMUTABLE, cannot change values in tuple
t[2]

# Sets
{1,2,3}
{1,2,3,2,3,2,1} # Unique elements only
s = {1,2,3}
s.add(5)

# Comparison Ops
1>2
1 == 2
1>2 and 2<3
1==1 or 2>3

if 1<2:
    print('ye')

if 1==2:
    print('1')
elif 2==3:
    print('2')
else:
    print('3')
    
#####################
###### Part 3 ######
#####################

# For loops
for i in range(1,6):
    print(i)
    
# While loops
x = 1
while x < 5:
    print(x)
    x+=1 # Prevents infinite looping
   
# List Comprehension
a = [1,2,3,4]
a2 = [i**2 for i in a]
a2

# Functions
def fun1(p1):
    print(p1)
    
fun1('Howdy')

def square(num):
    """
    THIS FUNCTION DOES A THING
    """
    return num**2
output = square(3)
output

#####################
###### Part 4 ######
#####################

# Lambda Expressions: Rewrite functions
seq = [1,2,3,4,5]
list(map(square, seq))
lambda num: num**2 # Pass into map() function
list(map(lambda num: num*2, seq))

# Filter
list(filter(lambda num: num % 2 == 0, seq)) # Even nums in seq

# Methods
s = 'This is a string with Things in it'
s.lower()
s.upper()
s.split()
s.split('T')

d.keys()
d.items()
d.values()

list1.pop()
list1
list1.append('d')
'a' in list1
'e' in list1

# Tuple unpacking
t = [(1,2), (3,4), (5,6)]
for a,b in t:
    print(a)
    print(b)