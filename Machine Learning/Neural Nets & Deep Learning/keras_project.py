# Tensorflow & Keras Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_info = pd.read_csv('DATA/lending_club_info.csv', index_col='LoanStatNew')

print(data_info.loc['revol_util']['Description'])


def lending_feature_info(col_name):
    print(data_info.loc[col_name]['Description'])


lending_feature_info('mort_acc')

df = pd.read_csv('DATA/lending_club_loan_two.csv')

df.info()

# EDA
sns.countplot(x='loan_status', data=df)

sns.histplot(data=df, x='loan_amnt', bins=40)

df.corr()

plt.figure(figsize=(12, 12))
sns.heatmap(data=df.corr(), cmap='plasma', annot=True)

lending_feature_info('installment')
lending_feature_info('loan_amnt')

sns.scatterplot(data=df, x='installment', y='loan_amnt')

sns.boxplot(data=df, x='loan_status', y='loan_amnt')

df.groupby('loan_status').describe()['loan_amnt']

df['grade'].unique()

df['sub_grade'].unique()

sns.countplot(data=df, x='grade', hue='loan_status')

plt.figure(figsize=(12, 8))
sns.countplot(data=df.sort_values('sub_grade'), x='sub_grade', palette='rocket')

plt.figure(figsize=(12, 8))
sns.countplot(data=df.sort_values('sub_grade'),
              x='sub_grade', hue='loan_status')

plt.figure(figsize=(12, 8))
sns.countplot(data=df.sort_values('sub_grade')[
              df['grade'].isin(['F', 'G'])], x='sub_grade', hue='loan_status')

df['loan_repaid'] = np.where(df['loan_status'] == 'Fully Paid', 1, 0)

df.head()

df.corr()['loan_repaid'][:-1].sort_values().plot(kind='bar')

# Preprocessing

len(df)

df.isna().sum()

(df.isna().sum() / len(df)) * 100

lending_feature_info('emp_title')
lending_feature_info('emp_length')

df['emp_title'].nunique()

df.drop('emp_title', inplace=True, axis=1)

df['emp_length'].unique()

year_categories = ['< 1 year', '1 year', '2 years', '3 years', '4 years',
                   '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']
df['emp_length'] = pd.Categorical(df['emp_length'], categories=year_categories)

plt.figure(figsize=(12, 8))
sns.countplot(x='emp_length', data=df.sort_values('emp_length'))

plt.figure(figsize=(12, 8))
sns.countplot(x='emp_length', data=df.sort_values(
    'emp_length'), hue='loan_status')

length_loan_count = df.groupby('emp_length')['loan_status'].count()

length_loan_paid = df.groupby('emp_length')['loan_repaid'].sum()

length_loan_pct = (length_loan_count - length_loan_paid) / length_loan_paid

plt.figure(figsize=(12, 8))
sns.barplot(x=length_loan_pct.index, y=length_loan_pct.values)

df.drop('emp_length', inplace=True, axis=1)

df.isna().sum()

df['title'].head(10)
df['purpose'].head(10)

df.drop('title', axis=1, inplace=True)

lending_feature_info('mort_acc')

df['mort_acc'].value_counts()

df.corr()['mort_acc']

total_acc_means = df.groupby('total_acc')['mort_acc'].mean()


def mort_acc_fill_value(total_acc, mort_acc):
    if (np.isnan(mort_acc)):
        return total_acc_means[total_acc]
    else:
        return mort_acc


df['mort_acc'] = df.apply(lambda x: mort_acc_fill_value(
    x['total_acc'], x['mort_acc']), axis=1)

df.isna().sum()

df.dropna(inplace=True)

df.isna().sum()

df.select_dtypes(exclude=['int32', 'float64']).columns

df['term'] = df['term'].apply(lambda x: x.strip() if isinstance(x, str) else x)
df['term'] = df['term'].map(lambda x: x.split(' ')[0])
df['term'] = pd.to_numeric(df['term'])

df.drop('grade', axis=1, inplace=True)

subgrade = pd.get_dummies(df['sub_grade'], drop_first=True)

df = pd.concat([df, subgrade], axis=1)

df.drop('sub_grade', axis=1, inplace=True)

df.columns

df.select_dtypes(exclude=['int32', 'int64', 'float64', 'uint8']).columns

verification = pd.get_dummies(df['verification_status'], drop_first=True)
app_type = pd.get_dummies(df['application_type'], drop_first=True)
init_list_status = pd.get_dummies(df['initial_list_status'], drop_first=True)
purpose = pd.get_dummies(df['purpose'], drop_first=True)

df = pd.concat([df, verification, app_type, init_list_status, purpose], axis=1)
df.drop(['verification_status', 'application_type',
        'initial_list_status', 'purpose'], axis=1, inplace=True)

df['home_ownership'].value_counts()

df['home_ownership'] = df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

home_ownership = pd.get_dummies(df['home_ownership'], drop_first=True)

df = pd.concat([df, home_ownership], axis=1)
df.drop('home_ownership', axis=1, inplace=True)

df['zip'] = df['address'].map(lambda x: x.split(' ')[-1])

zip_code = pd.get_dummies(df['zip'], drop_first=True)

df = pd.concat([df, zip_code], axis=1)

df.drop(['zip', 'address'], axis=1, inplace=True)

df.drop('issue_d', axis=1, inplace=True)

df['earliest_cr_year'] = df['earliest_cr_line'].apply(
    lambda x: pd.to_numeric(x.split('-')[1]))

df.drop('earliest_cr_line', axis=1, inplace=True)
