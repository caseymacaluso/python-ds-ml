# Linear Regression Project

# Library Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data Import
ecomm = pd.read_csv("Ecommerce Customers.csv")

# Inspecting data
ecomm.head()
ecomm.info()
ecomm.describe()

# Checking if there's any correlation between time spent on app or website vs amount spent
sns.jointplot(ecomm['Time on Website'], ecomm['Yearly Amount Spent'])
sns.jointplot(ecomm['Time on App'], ecomm['Yearly Amount Spent'])
sns.jointplot(ecomm['Time on App'], ecomm['Length of Membership'], kind="hex")

# Yearly Amount spent seems to correlate with Length of Membership
sns.pairplot(ecomm)
sns.lmplot(data=ecomm, y="Yearly Amount Spent", x="Length of Membership")

# Set up the data we want to use for our model
ecomm.columns
x = ecomm[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = ecomm['Yearly Amount Spent']

# Split our data into training and testing data. Using 30% for our training set portion
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

# Define our linear model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()

# Fit the linear model to our training data
lm.fit(x_train, y_train)
# Show our model's coefficients
print(lm.coef_)

# Make predictions with our model using the testing data
predictions = lm.predict(x_test)

# Plot our predictions against the testing data
# Results look very promising here, very strong correlation observed here
plt.scatter(y_test, predictions)

# Regression Metrics
from sklearn import metrics

print("MAE: ", metrics.mean_absolute_error(y_test, predictions))
print("MSE: ", metrics.mean_squared_error(y_test, predictions))
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print("R^2: ", metrics.explained_variance_score(y_test, predictions)) # 0.989, very good representation

# Residuals are normally distributed, so things look good here as well.
sns.distplot((y_test - predictions), bins=50)

# Creating a dataframe of the coefficients
cdf = pd.DataFrame(lm.coef_, x.columns, columns=["Coefficient"])
#                       Coefficient
# Avg. Session Length     25.981550
# Time on App             38.590159
# Time on Website          0.190405
# Length of Membership    61.279097

# Based on this model, the app currently provides a better return than the website. 1 minute spent on the app = ~$39 total amount spent
# There are other factors to consider here, such as cost to invest in the app vs the website, as well as how length of membership plays in to
# total amount spent. Ultimately, it will be up to the business to decide if they want to fully invest in the app or try to come up with ways to
# make the website more profitable.