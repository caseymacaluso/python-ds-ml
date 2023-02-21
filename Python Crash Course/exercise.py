# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 19:50:33 2021

@author: Casey Macaluso
"""

# 7 to 4th power
7**4

# String splitting
s = 'Hello there Casey!'
s.split()

# String print formatting
p = "Earth"
diam = 12742
print('The diameter of {one} is {two} kilometers.'.format(one=p, two=diam))

# Nested list extraction
lst = [1,2,[3,4],[5,[100,200,['hello']],23,11],1,7]
lst[3][1][2][0]

# Nested dictionary extraction
d = {'k1':[1,2,3,{'tricky':['oh','man','inception',{'target':[1,2,3,'hello']}]}]}
d['k1'][3]['tricky'][3]['target'][3]

# Function for extracting email domain
def email_dom(email):
    return email.split('@')[1]
email_dom('user@domain.com')

# Function for dog finding
def findDog(sentence):
    if 'dog' in sentence:
        return True
    else:
        return False
findDog('Is there a dog here?')
findDog('There is a canine here.')

# Count number of dogs
def countDog(sentence):
    count = 0
    seq = sentence.split()
    for i in seq:
        if i == 'dog':
            count+=1
    return count

countDog('There is dog here and a dog there')

# Lambda Exp
seq = ['soup','dog','salad','cat','great']
list(filter(lambda item: item[0] == 's', seq))

# Functions
def caught_speeding(speed, birthday):
    if birthday == False: 
        if speed <= 60:
            return 'No Ticket'
        elif (speed > 60 and speed <= 80):
            return 'Small Ticket'
        else:
            return 'Big Ticket'
    else:
        if speed <= 65:
            return 'No Ticket'
        elif (speed > 65 and speed <= 85):
            return 'Small Ticket'
        else:
            return 'Big Ticket'

caught_speeding(81, True)
caught_speeding(81, False)
