# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn. 
4. Assign the points for representing in the graph. 
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: MADHUVATHANI.V 
RegisterNumber:  212223040107

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print("df.head\n",df.head())
print("df.tail\n",df.tail())
x = df.iloc[:,:-1].values
print("Array value of x:",x)
y = df.iloc[:,1].values
print("Array value of y:",y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print("Values of y predict:\n",y_pred)
print("Array values of y test:\n",y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:

# df.head

![420577359-dc344a5a-cedd-4007-be03-9fcb81134fe7](https://github.com/user-attachments/assets/10c3e005-d427-43a6-a933-e5a8d6c28cca)

# df.tail

![420577385-1a897782-448a-4ff1-8217-000b2b5ef6db](https://github.com/user-attachments/assets/5c2e7b72-59df-42d9-a86f-46678a4c94d0)

# Array value of X

![420577419-033fad71-bfa7-4506-b3ff-746563894a0a](https://github.com/user-attachments/assets/feafb843-0b36-40bc-aab8-8891024f4132)

# Array value of Y

![420577455-3ab1caf4-b899-4d43-9386-e465bffef739](https://github.com/user-attachments/assets/11b6265c-d680-449b-9c0e-840d1be6a79e)

# Values of Y prediction

![420577475-654bf50e-8be0-4215-b333-19f703613c93](https://github.com/user-attachments/assets/293c6063-1e87-4475-9526-867db9dfa343)

# Array values of Y test

![420577504-066572fd-90e8-4879-b0f0-6dc3a01092a5](https://github.com/user-attachments/assets/18d0c77a-5863-4ca1-8dbe-b40ef8be0235)

# Training Set Graph

![420577556-1454169e-4d85-49f3-8697-7ee4c801de7b](https://github.com/user-attachments/assets/2fee8219-fe04-4a8e-a3ac-1d54b5a0b659)


# Training Set Graph

![420577589-be1543ca-4f1a-4584-8d5b-5c671ca11aae](https://github.com/user-attachments/assets/38bf7480-d807-4d86-af3d-28e2e40ced7c)

# Values of MSE, MAE and RMSE

![420577623-8f2f8312-52a6-44a5-962c-c5cea9de2155](https://github.com/user-attachments/assets/7e3ea3fd-5de6-4256-ba81-9325d36d99e1)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
