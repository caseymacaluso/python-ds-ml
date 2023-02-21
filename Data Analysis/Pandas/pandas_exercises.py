# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 23:30:56 2021

@author: Casey Macaluso
"""

import pandas as pd
import os

os.chdir('Data Analysis/Pandas')
sal = pd.DataFrame(pd.read_csv('Salaries.csv'))

sal.head()
sal.info()

### SF SALARIES ###
sal['BasePay'].mean()
sal['OvertimePay'].max()
sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['JobTitle']
sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['TotalPayBenefits']
sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].max()]
# OR
sal.sort_values(by='TotalPayBenefits', ascending=False).head(1)

sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].min()]['TotalPayBenefits']
# OR
sal.sort_values(by='TotalPayBenefits').head(1)['TotalPayBenefits'] # They are losing money

yr_grp = sal.groupby('Year')
yr_grp['BasePay'].mean()
sal['JobTitle'].nunique()
sal['JobTitle'].value_counts().head()
sum(sal[sal['Year'] == 2013]['JobTitle'].value_counts() == 1)
# OR
yr2013 = sal[sal['Year'] == 2013]
len(yr2013['JobTitle'].value_counts()[yr2013['JobTitle'].value_counts() == 1])

def chief(title):
    if 'chief' in title.lower().split():
        return True
    else:
        return False
sum(sal['JobTitle'].apply(lambda x: chief(x)))

sal_copy = sal.copy()
sal_copy['title_len'] = sal_copy['JobTitle'].str.len()
sal_corr = sal_copy[['title_len', 'TotalPayBenefits']]
sal_corr.corr()

### ECOMM PURCHASES ###
ecom = pd.DataFrame(pd.read_csv('Ecommerce Purchases'))
ecom.head()
ecom.info()
ecom['Purchase Price'].mean()
ecom['Purchase Price'].max()
ecom['Purchase Price'].min()
len(ecom[ecom['Language'] == 'en'])
len(ecom[ecom['Job'] == 'Lawyer'])
ecom['AM or PM'].value_counts()
ecom['Job'].value_counts().head()
ecom[ecom['Lot'] == '90 WT']['Purchase Price']
ecom[ecom['Credit Card'] == 4926535242672853]['Email']
len(ecom[(ecom['CC Provider'] == 'American Express') & (ecom['Purchase Price'] > 95)])

sum(ecom['CC Exp Date'].apply(lambda cc: cc.split('/')[1] == '25'))
ecom['Email'].apply(lambda email: email.split('@')[1]).value_counts().head()
