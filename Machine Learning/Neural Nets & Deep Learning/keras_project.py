# Tensorflow & Keras Project

# Library Imports
import random
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data description import, set statistic to index
data_info = pd.read_csv('DATA/lending_club_info.csv', index_col='LoanStatNew')

# Example, print description for 'revol_util'
print(data_info.loc['revol_util']['Description'])

# Function to print description for loan statistic


def lending_feature_info(col_name):
    print(data_info.loc[col_name]['Description'])


# Using function for 'mort_acc'
lending_feature_info('mort_acc')

# Actual data import
df = pd.read_csv('DATA/lending_club_loan_two.csv')

# Several columns with nulls, we'll explore these later
df.info()

# EDA

# Countplot showing # of borrowers who've paid off loans vs not fully paid
sns.countplot(x='loan_status', data=df)

# Distribution of the loan amount
sns.histplot(data=df, x='loan_amnt', bins=40)

# Correlation of all numeric information
df.corr()

# Heatmap of correlated data
plt.figure(figsize=(12, 12))
sns.heatmap(data=df.corr(), cmap='plasma', annot=True)

# High correlation between 'installment' and 'loan_amnt'
# Descriptions for 'installment' and 'loan_amnt'
lending_feature_info('installment')
lending_feature_info('loan_amnt')

# Scatterplot of 'installment' vs. 'loan_amnt"
sns.scatterplot(data=df, x='installment', y='loan_amnt')
# Makes sense that these are related, likely some internal calculation to find monthly installment based on the total loan amount

# Boxplot of 'loan_status' vs. 'loan_amnt'
sns.boxplot(data=df, x='loan_status', y='loan_amnt')

# Loan amount stats per loan status group
df.groupby('loan_status').describe()['loan_amnt']

# Find unique loan grades
df['grade'].unique()

# Find unique loan subgrades
df['sub_grade'].unique()

# Countplot counting number of grades, split by loan status
sns.countplot(data=df, x='grade', hue='loan_status')

# Countplot counting number of subgrades
plt.figure(figsize=(12, 8))
sns.countplot(data=df.sort_values('sub_grade'), x='sub_grade', palette='rocket')

# Countplot counting number of subgrades, split by loan status
plt.figure(figsize=(12, 8))
sns.countplot(data=df.sort_values('sub_grade'),
              x='sub_grade', hue='loan_status')
# Appears to be relatively even for F and G subgrades, so let's isolate those and see what we can find out

# Countplot with F and G subgrades, split by loan status
plt.figure(figsize=(12, 8))
sns.countplot(data=df.sort_values('sub_grade')[
              df['grade'].isin(['F', 'G'])], x='sub_grade', hue='loan_status')

# Adding dummy variable for our model, based on value for 'loan_status
df['loan_repaid'] = np.where(df['loan_status'] == 'Fully Paid', 1, 0)

df.head()

# Plotting the data correlated with our new 'loan_repaid' column
df.corr()['loan_repaid'][:-1].sort_values().plot(kind='bar')

# Preprocessing

len(df)

# Checking number of null values per column
df.isna().sum()

# Showing nulls per column as percentage of the dataset
(df.isna().sum() / len(df)) * 100

# Let's check 'emp_title' and 'emp_length' first
lending_feature_info('emp_title')
lending_feature_info('emp_length')

# Number of unique emp titles
df['emp_title'].nunique()

# We have over 170k unique titles, so we can't adequately use these titles in our model. We'll just drop this column
df.drop('emp_title', inplace=True, axis=1)

# Unique employment lengths
df['emp_length'].unique()

# Setting our 'emp_length' as a category that we can sort on
year_categories = ['< 1 year', '1 year', '2 years', '3 years', '4 years',
                   '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']
df['emp_length'] = pd.Categorical(df['emp_length'], categories=year_categories)

# Countplot showing the number of loans whose borrowers have employment lengths in different year categories
plt.figure(figsize=(12, 8))
sns.countplot(x='emp_length', data=df.sort_values('emp_length'))

# Similar to above, but splitting based on loan_status
plt.figure(figsize=(12, 8))
sns.countplot(x='emp_length', data=df.sort_values(
    'emp_length'), hue='loan_status')

# Calculating the percentage of loans that don't get paid off per emp_length group
length_loan_count = df.groupby('emp_length')['loan_repaid'].count()

length_loan_paid = df.groupby('emp_length')['loan_repaid'].sum()

length_loan_pct = (length_loan_count - length_loan_paid) / length_loan_paid

plt.figure(figsize=(12, 8))
sns.barplot(x=length_loan_pct.index, y=length_loan_pct.values)

# Amount that does not get paid off per group is roughly the same, so this column won't provide much value for our analysis
df.drop('emp_length', inplace=True, axis=1)

# Checking nulls again
df.isna().sum()

# Let's check 'title' and 'purpose' to see how those two relate
df['title'].head(10)
df['purpose'].head(10)

# Title is essentially an extension of the purpose column, so we can drop this as well
df.drop('title', axis=1, inplace=True)

# Now we look at 'mort_acc' which has the most remaining nulls
lending_feature_info('mort_acc')

df['mort_acc'].value_counts()

# Checking if any of our existing columns correlate with the 'mort_acc' column
df.corr()['mort_acc']

# The 'total_acc' column seems to have a solid correlation, so let's find the average of the mort_acc column for each group of 'total_acc'
total_acc_means = df.groupby('total_acc')['mort_acc'].mean()

# Function to fill null values in the 'mort_acc' column with the 'mort_acc' average based on how many total accounts the borrower has


def mort_acc_fill_value(total_acc, mort_acc):
    if (np.isnan(mort_acc)):
        return total_acc_means[total_acc]
    else:
        return mort_acc


# Apply our new function to the 'mort_acc' column
df['mort_acc'] = df.apply(lambda x: mort_acc_fill_value(
    x['total_acc'], x['mort_acc']), axis=1)

# Checking nulls again
df.isna().sum()

# Remaining nulls are very small in scale, so we'll just drop the remaining rows with null values
df.dropna(inplace=True)

df.isna().sum()

# Nulls look good, now we need to convert any string/categorical items to numerical for our model
# Checking which columns we need to change
df.select_dtypes(exclude=['int32', 'float64']).columns

# First, removing leading and trailing spaces from 'term' column
df['term'] = df['term'].apply(lambda x: x.strip() if isinstance(x, str) else x)
# Now, we split based on the space and grab the number
df['term'] = df['term'].map(lambda x: x.split(' ')[0])
# Finally, convert to numeric
df['term'] = pd.to_numeric(df['term'])

# Dropping grade, as subgrade includes the grade
df.drop('grade', axis=1, inplace=True)

# Dummies for the 'subgrade' column
subgrade = pd.get_dummies(df['sub_grade'], drop_first=True)

# Concatenate to df, then drop original column
df = pd.concat([df, subgrade], axis=1)
df.drop('sub_grade', axis=1, inplace=True)

df.columns
df.select_dtypes(exclude=['int32', 'int64', 'float64', 'uint8']).columns

# Dummy data for more categorical columns
verification = pd.get_dummies(df['verification_status'], drop_first=True)
app_type = pd.get_dummies(df['application_type'], drop_first=True)
init_list_status = pd.get_dummies(df['initial_list_status'], drop_first=True)
purpose = pd.get_dummies(df['purpose'], drop_first=True)

# Concatenate to df and drop original columns
df = pd.concat([df, verification, app_type, init_list_status, purpose], axis=1)
df.drop(['verification_status', 'application_type',
        'initial_list_status', 'purpose'], axis=1, inplace=True)

# Checking values for home_ownership
df['home_ownership'].value_counts()

# Modifying column to group 'NONE' and 'ANY' values with 'OTHER' (i.e. grouping all three into one column, as it's very small compared to the others)
df['home_ownership'] = df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

# Dummies for home_ownership
home_ownership = pd.get_dummies(df['home_ownership'], drop_first=True)

# Concatenate and drop original
df = pd.concat([df, home_ownership], axis=1)
df.drop('home_ownership', axis=1, inplace=True)

# Grabbing zip code from address
df['zip'] = df['address'].map(lambda x: x.split(' ')[-1])

# Dummies for zip code
zip_code = pd.get_dummies(df['zip'], drop_first=True)

# Concatenate and drop original
df = pd.concat([df, zip_code], axis=1)
df.drop(['zip', 'address'], axis=1, inplace=True)
# Dropping issue date, not useful here
df.drop('issue_d', axis=1, inplace=True)

# Grabbing the year of when the borrower's credit line was opened
df['earliest_cr_year'] = df['earliest_cr_line'].apply(
    lambda x: pd.to_numeric(x.split('-')[1]))

# Dropping earliest_cr_line and original loan_status columns
df.drop('earliest_cr_line', axis=1, inplace=True)
df.drop('loan_status', axis=1, inplace=True)

# Data Preprocessing done! Now let's get into model creation and evaluation

# TF Model Creation

# Training and Testing Data
X = df.drop('loan_repaid', axis=1).values
y = df['loan_repaid'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101)

# Scaler to scale our data appropriately
scaler = MinMaxScaler()

# Transform & fit training data, transform testing data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Definition
model = Sequential()
model.add(Dense(78, activation='relu'))
model.add(Dense(39, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

# Fitting model to training data, passing in testing data for validation
model.fit(x=X_train, y=y_train, epochs=25, validation_data=(
    X_test, y_test), verbose=1, batch_size=256)

# Saving model for future use
model.save('loan_repayment_model.h5')

# Saving losses and plotting
losses = pd.DataFrame(model.history.history)
losses.plot()

# Making predictions based on testing data
predictions = (model.predict(X_test) > 0.5)*1

# Metrics
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

#               precision    recall  f1-score   support

#            0       0.91      0.48      0.63     15658
#            1       0.88      0.99      0.93     63386

#     accuracy                           0.89     79044
#    macro avg       0.90      0.73      0.78     79044
# weighted avg       0.89      0.89      0.87     79044

# [[ 7463  8195]
#  [  749 62637]]

# Testing model with random customer
random.seed(101)
random_ind = random.randint(0, len(df))
new_customer = df.drop('loan_repaid', axis=1).iloc[random_ind]
new_customer

# Reshaping customer to work with the model
(model.predict(new_customer.values.reshape(1, 78)) > 0.5)*1
# array([[1]])

# Checking actual value to see if we predicted correctly
df.iloc[random_ind]['loan_repaid']
# 1.0
