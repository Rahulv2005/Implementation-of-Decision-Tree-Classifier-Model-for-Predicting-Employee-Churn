# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.
2.Upload and read the dataset.
3.Check for any null values using the isnull() function.
4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: rahul V
RegisterNumber:  212223240132
*/
```
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
data=pd.read_csv("Employee_EX6.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
plt.figure(figsize=(18,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()

## Output:
![decision tree classifier model](sam.png)

![WhatsApp Image 2024-04-05 at 10 24 13_494c920e ml 01](https://github.com/Rahulv2005/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/152600335/33282932-4638-4670-b8a0-f69ec5b353e2)

![WhatsApp Image 2024-04-05 at ml 02](https://github.com/Rahulv2005/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/152600335/1cc2e472-6553-4456-aa72-f32983ff553d)

![WhatsApp Image 2024-04-05 at 10 24 49_bda2dc1f ml 03](https://github.com/Rahulv2005/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/152600335/5cf17138-a111-4752-b3ed-648e6889a341)

![WhatsApp Image 2024-04-05 at 10 25 11_df075322 ml 04](https://github.com/Rahulv2005/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/152600335/3fb7ffa4-d60c-4545-955a-2a024b823881)






## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
