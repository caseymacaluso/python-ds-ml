# Linear Regression Notes

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read in USA housing data
df = pd.read_csv("USA_Housing.csv")

# Inspecting the data
df.head()
df.info()
df.describe()

# Plotting data to gain an understanding for it
sns.pairplot(df)
sns.distplot(df['Price'])
sns.heatmap(df.corr(), annot=True)

# Listing column names, want to exclude 'Address' from our model
df.columns

# Defining our  x and y for our model
x = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
y = df['Price']

# Importing train_test_split to separate our dataset into training and testing data
# Setting test size to be 40% of our dataset, could also use 30%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=101)

# Define our linear model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()

# Fit the linear model to our training data
lm.fit(x_train, y_train)

# Looking at intercepts and coefficients
print(lm.intercept_)
lm.coef_
# Making a df of our coefficients to better interpret what they mean
# i.e 1 unit increase in avg area income = ~$21 increase in price
cdf = pd.DataFrame(lm.coef_,x.columns, columns=['Coefficient'])

# Make predictions with our model using the testing data
predictions = lm.predict(x_test)

# Plotting our predictions against our y_test data
plt.scatter(y_test, predictions)

# Making a histograms of our residuals
sns.distplot((y_test - predictions), bins=50)
# Our residuals look normally distributed, which is a good sign for our model

# Regression Evaluation Metrics
from sklearn import metrics

print("MAE: ", metrics.mean_absolute_error(y_test, predictions))
print("MSE: ", metrics.mean_squared_error(y_test, predictions))
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, predictions)))
