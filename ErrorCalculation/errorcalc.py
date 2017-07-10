import pandas as pd 
from sklearn import linear_model 
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 

#reading data from dataset : 

dataframe = pd.read_csv('challenge_dataset.txt', header = 0, names=["x", "y"])

print dataframe

x_values = dataframe[['x']] 
y_values = dataframe[['y']] 

body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

x = dataframe['x'].values 
y = dataframe['y'].values

dataset_error = mean_squared_error(x, y)
prediction_error = mean_squared_error(x_values, body_reg.predict(y_values))


print "Error of Dataset: " 
print dataset_error 

print "Error of Prediction: "
print prediction_error 

print "Difference: "
print prediction_error - dataset_error

plt.scatter(x_values, y_values) 
plt.plot(x_values, body_reg.predict(y_values)) 
plt.show()
