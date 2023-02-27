# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 23:30:56 2021

@author: Casey Macaluso
"""

import pandas as pd
import os

os.chdir('Data Analysis/Pandas')
sal = pd.DataFrame(pd.read_csv('Salaries.csv'))

# Check head of the data
sal.head()

# Check the data for how many entries there are
sal.info()

###################
### SF SALARIES ###
###################

# Average pase pay
sal['BasePay'].mean()

# Highest overtime pay
sal['OvertimePay'].max()

# Job title of Joseph Driscoll
sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['JobTitle']

# How much does Joseph Driscoll make
sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['TotalPayBenefits']

# Who makes the most
sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].max()]
# OR
sal.sort_values(by='TotalPayBenefits', ascending=False).head(1)

# Who makes the least
sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].min()]['TotalPayBenefits']
# OR
sal.sort_values(by='TotalPayBenefits').head(1)['TotalPayBenefits'] # They are losing money

# Average base pay of all employees per pear
yr_grp = sal.groupby('Year')
yr_grp['BasePay'].mean()

# Number of unique job titles
sal['JobTitle'].nunique()

# Most frequent job titles
sal['JobTitle'].value_counts().head()

# Job titles with only one occurrence in 2013
sum(sal[sal['Year'] == 2013]['JobTitle'].value_counts() == 1)
# OR
yr2013 = sal[sal['Year'] == 2013]
len(yr2013['JobTitle'].value_counts()[yr2013['JobTitle'].value_counts() == 1])

# Function to see how many people have 'Chief' in their job title
def chief(title):
    if 'chief' in title.lower().split():
        return True
    else:
        return False
sum(sal['JobTitle'].apply(lambda x: chief(x)))

# Checking for correlation between the length of the 'JobTitle' string and salary
sal_copy = sal.copy()
sal_copy['title_len'] = sal_copy['JobTitle'].str.len()
sal_corr = sal_copy[['title_len', 'TotalPayBenefits']]
sal_corr.corr()

#######################
### ECOMM PURCHASES ###
#######################

ecom = pd.DataFrame(pd.read_csv('Ecommerce Purchases'))

# Check the head of the ecomm data
ecom.head()

# Check for more information on the dataframe
ecom.info()

# Average purchase price
ecom['Purchase Price'].mean()

# Highest purchase price
ecom['Purchase Price'].max()

# Lowest purchase price
ecom['Purchase Price'].min()

# Number of people with English language set as the language on the website
len(ecom[ecom['Language'] == 'en'])

# Number of people where the occupation was Lawyer
len(ecom[ecom['Job'] == 'Lawyer'])

# Number of transactions for morning and evening
ecom['AM or PM'].value_counts()

# Most common jobs titles
ecom['Job'].value_counts().head()

# Purchase price of a purchase made from Lot 90 WT
ecom[ecom['Lot'] == '90 WT']['Purchase Price']

# Email address listed for customer with credit card 4926535242672853
ecom[ecom['Credit Card'] == 4926535242672853]['Email']

# Number of people who have an American Express card and whose purchase price was over $95
len(ecom[(ecom['CC Provider'] == 'American Express') & (ecom['Purchase Price'] > 95)])

# Number of people whose credit cards expire in 2025
sum(ecom['CC Exp Date'].apply(lambda cc: cc.split('/')[1] == '25'))

# Most common email providers among customers
ecom['Email'].apply(lambda email: email.split('@')[1]).value_counts().head()
