# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### 1.Import the standard Libraries. 
#### 2.Set variables for assigning dataset values. 
#### 3.Import linear regression from sklearn. 
#### 4.Assign the points for representing in the graph. 
#### 5.Predict the regression for marks by using the representation of the graph. 
#### 6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: ABISHEIK RAJ.J
RegisterNumber:  212224230006
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
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
## DATASET
![Screenshot 2025-03-13 091742](https://github.com/user-attachments/assets/43bd929c-7c03-43c9-8b32-890fd1105d1c)
## HEAD VALUE
![Screenshot 2025-03-13 091803](https://github.com/user-attachments/assets/81c4c897-249f-4634-83ba-a947c3026910)

## TAIL VALUE
![Screenshot 2025-03-13 091829](https://github.com/user-attachments/assets/f9d38229-f1b0-4a85-bb36-b12136111f27)
## X AND Y VALUE
![Screenshot 2025-03-13 091906](https://github.com/user-attachments/assets/732c844c-d034-428a-93d1-af9a6751cc66)

## PREDICATION OF VALUE X AND Y
![Screenshot 2025-03-13 091954](https://github.com/user-attachments/assets/fb4289f4-3e9f-4e0e-ad09-f97c1838a55a)
## MSE,MAE,RMSE
![Screenshot 2025-03-13 092109](https://github.com/user-attachments/assets/0cfe3086-0e92-439e-af2c-9291670881ff)

## TRAINING SET
![Screenshot 2025-03-13 092041](https://github.com/user-attachments/assets/18fdb7d1-1e1d-44c6-aa68-484a08a43b7a)

## TESTING SET
![Screenshot 2025-03-13 092102](https://github.com/user-attachments/assets/0451e91f-a7d5-4ecd-aa6a-30ce2e5110c6)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
