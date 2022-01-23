import numpy as pynum
import csv
import pandas as pd
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# reads the csv file
df = pd.read_csv('Salary_Data.csv')

# gets the first 5 rows
#print(df.head())

# prints the data types of each column
#print(df.dtypes)

#removing dupliactes from the database
df.drop_duplicates()

#printing the dimension of the database
#print(df.shape)

#checking for NULL values in the database and summing them from each column
# this ensures that all NULL values are removed
#print(df.isnull().sum())

#how to creating dependent and independent variables

# this actually needs to be one of the columns from the database
target_feature = 'Salary'

# creating the dependent variable
y = df[target_feature]

#creating the independent variable
X = df.drop(target_feature, axis=1)


#visualizing the data by a scatter plot
plot.scatter(X,y)
plot.xlabel("YearsExperience")
plot.ylabel('Salary')
plot.grid()
#plot.show()

#selecting random samples
train_x, test_x, train_y, test_y = train_test_split(X,y, test_size = 0.2)
'''
train_x = X[:80]
train_y = y[:80]

test_x = X[80:]
test_y = y[80:] '''

# training our model
regr = LinearRegression()

regr.fit(train_x, train_y)

#comparing accuraccy of results
regr.score(train_x, train_y)

#predicting y-values based on x-test samples
y_pred = regr.predict(test_x)
df1 = pd.DataFrame({'Actual': test_y, 'Predicted': y_pred, 'variance': test_y-y_pred})
print(df1.head())

#visualizing the traisning set results
plot.scatter(train_x, train_y, color='red')
plot.plot(train_x, regr.predict(train_x), color='blue')
plot.title("Salary vs Experience")
plot.xlabel("Year of expeirence")
plot.ylabel("Salary")
plot.grid()
plot.show()

#visualizing the testing set results
plot.scatter(test_x, test_y, color='red')
plot.plot(train_x, regr.predict(train_x), color='blue')
plot.title("Salary vs Experience")
plot.xlabel("Year of expeirence")
plot.ylabel("Salary")
plot.grid()
plot.show()

print(metrics.mean_absolute_error(test_y, y_pred))

print(metrics.mean_squared_error(test_y, y_pred))

#regr.intercept
#regr.coef