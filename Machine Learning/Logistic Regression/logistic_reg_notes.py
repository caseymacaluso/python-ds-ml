# Logistic Regression Notes

# Library Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('titanic_train.csv')
train.head()

# Getting a view for which columns have null values
sns.heatmap(train.isnull(), cbar=False, yticklabels=False, cmap='viridis')
sns.set_style('whitegrid')

# Some exploratory DA
# > 500 passengers ended up dying
sns.countplot(x='Survived', data=train)
# Of the passengers that died, the vast majority were males
sns.countplot(x='Survived', hue='Sex', data=train)
# Of the passengers that died, most were in the 3rd (lowest) class
sns.countplot(x='Survived', hue='Pclass', data=train)

# Couple of bins for age, one ranging from 0-10, and another from ~25-40
sns.distplot(train['Age'].dropna(), kde=False, bins=30)

# Many on the trip had no siblings or spouses
sns.countplot(x='SibSp', data=train)

# Many passengers paid for very cheap fare
train['Fare'].hist(bins=40, figsize=(10,4))

# Looking at each class' age distribution
plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass', y='Age', data=train, palette='winter')

# Calculating average ages for each passenger class
class1_age = np.floor(np.mean(train[train['Pclass'] == 1]['Age']))
class2_age = np.floor(np.mean(train[train['Pclass'] == 2]['Age']))
class3_age = np.floor(np.mean(train[train['Pclass'] == 3]['Age']))

# Function to impute missing age values based on class
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return class1_age
        elif Pclass == 2:
            return class2_age
        else:
            return class3_age
    else:
        return Age
    
# Imputing age value
train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)
train.info()

# Nulls for 'Age' are done, now we need to address 'Cabin' and one other outlier
sns.heatmap(train.isnull(), cbar=False, yticklabels=False, cmap='viridis')

# Dropping 'Cabin' column, as there isn't much of value to learn from here
train.drop('Cabin', axis=1, inplace=True)
# Dropping final record with null value
train.dropna(inplace=True)

# Nulls are taken care of, now we can move on
sns.heatmap(train.isnull(), cbar=False, yticklabels=False, cmap='viridis')

# Converting fields that can be looked at as categories into dummy columns (0-1)
# Dropping one column from this to avoid multicolinearity
sex = pd.get_dummies(train['Sex'], drop_first=True)
# Embark = where did the passenger embark from, has three choices, so dropping one to avoid multicolinearity
embark = pd.get_dummies(train['Embarked'], drop_first=True)
train = pd.concat([train, sex, embark], axis=1)
# Dropping original columns that have been dummy-fied, and any other columns that don't provide much value (yet)
train.drop(['Sex', 'Embarked', 'Ticket', 'Name'], axis=1, inplace=True)
train.drop('PassengerId', axis=1, inplace=True)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train.drop('Survived', axis=1), train['Survived'], test_size=0.3, random_state=101)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(solver="liblinear")
logmodel.fit(x_train, y_train)

predictions = logmodel.predict(x_test)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, predictions)
# [[148,  15],
#  [ 36,  68]]

print(classification_report(y_test, predictions))

#               precision    recall  f1-score   support

#            0       0.80      0.91      0.85       163
#            1       0.82      0.65      0.73       104

#     accuracy                           0.81       267
#    macro avg       0.81      0.78      0.79       267
# weighted avg       0.81      0.81      0.80       267
