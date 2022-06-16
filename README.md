# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1.Use the standard libraries in python for finding linear regression.
2.Set variables for assigning dataset values
3.Import linear regression from sklearn.
4.Predict the values of array.
5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: V.CHARAN SAI
RegisterNumber:  212221240061
*/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
datasets=pd.read_csv('Social_Network_Ads (1).csv')
x=datasets.iloc[:,[2,3]].values
y=datasets.iloc[:,4].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_x
StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_Test=sc_x.transform(x_test)
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
y_pred
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm
from sklearn import metrics
accuracy =metrics.accuracy_score(y_test,y_pred)
accuracy
recall_sensitivity=metrics.recall_score(y_test,y_pred,pos_label=1)
recall_specificity=metrics.recall_score(y_test,y_pred,pos_label=0)
recall_sensitivity,recall_specificity
from matplotlib.colors import ListedColormap
x_Set,y_Set=x_train,y_train
x1,x2=np.meshgrid(np.arange(start=x_Set[:,0].min()-1,stop=x_Set[:,0].max()+1,step=0.01),np.arange(start=x_Set[:,1].min()-1,stop=x_Set[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x2.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_Set)):
  plt.scatter(x_Set[y_Set==j,0],x_Set[y_Set==j,1],c=ListedColormap(('blue','black'))(i),label=j)
  plt.title('LogisticRegression(Trainingset)')
  plt.xlabel('Age')
  plt.ylabel('Estimated Salary')
  plt.legend()
  plt.show()


```

## Output:
<img width="401" alt="AAAAA" src="https://user-images.githubusercontent.com/94296221/174051893-40560126-7d3f-469e-851c-d23648d1c37a.png">
<img width="395" alt="BBBBBBBB" src="https://user-images.githubusercontent.com/94296221/174051903-a48b32b7-3939-43b6-afb4-2a21d80433c4.png">



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

