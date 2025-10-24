# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
#Load Dataset
dataset= pd.read_csv("Salary_Data.csv")
#Remove any rows that contain NaN values
dataset=dataset.dropna()

#Split the data(input/output)
#x= dataset.iloc [:,[]].values
x=dataset["Years of Experience"].values
y=dataset["Salary"].values

#Reshape x → 2D
x = x.reshape(-1, 1)

#Split the data ( into 70% training and 30% testing)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=20)
#Train the Linear Regression model
from sklearn.linear_model import LinearRegression
model_regression =LinearRegression()
model_regression.fit(x_train, y_train)

#predictions
y_pred=model_regression.predict(x_test)

#Evaluate the model
from sklearn.metrics import mean_absolute_error
print ('mean_absolute_error =',mean_absolute_error(y_test, y_pred))
######################################
A=model_regression.coef_
B=model_regression.intercept_

print ("y=",A,"x+",B)
##################################
#to know salary of a an employee with 10.5 experience 

print(model_regression.predict([[10.5]]))

#visualization__train. 
import matplotlib.pyplot as plt
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train,model_regression.predict(x_train) ,color='blue')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("the Years of Experience vs salary ")
plt.show()

#visualization__test.
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train,model_regression.predict(x_train) ,color='blue')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("the Years of Experience vs salary in testing")
plt.show()
