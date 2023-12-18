import pandas as pd

data = pd.read_csv(r'C:\Users\FCT\Desktop\Final year project\Churn_Modelling1 .csv')
print("Top  5 Dataset")
data.head()
print("Last 5 Dataset")
data.tail()
print("Number of Rows",data.shape[0])
print("Number of Columns",data.shape[1])
print("Information about our Dataset")
data.info()
print("Checking Null value in our Dataset")
data.isnull().sum()
print("Overall Statistics of Data")
data.describe(include ='all')
print('Dropping Irrelevant Features')
data.columns
data=data.drop(['RowNumber', 'CustomerId', 'Surname'],axis=1)
print('Encoding Categorical Data')
data['Geography'].unique()
data=pd.get_dummies(data,drop_first=True)
data.head(10)
data['Exited'].value_counts()

import seaborn as sns

sns.countplot(data['Exited'])
X= data.drop('Exited',axis=1)
y=data['Exited']
print("Handling Imbalanced Data With SMOTE")

from imblearn.over_sampling import SMOTE

X_res, y_res = SMOTE().fit_resample(X, y)
y_res.value_counts()
print("spllitig the Dataset")

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X_res, y_res,test_size=0.20,random_state=42)
print("Feature Scaling")

from sklearn.preprocessing import StandardScaler

sc= StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
X_train

from sklearn.linear_model import LogisticRegression

log = LogisticRegression()
log.fit(X_train, y_train)
y_pred1= log.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred1)
accuracy_score(y_test,y_pred1)

from sklearn.metrics import precision_score, recall_score, f1_score

precision_score(y_test,y_pred1)
precision_score(y_test,y_pred1)
recall_score(y_test,y_pred1)
recall_score(y_test,y_pred1)
f1_score(y_test,y_pred1)
f1_score(y_test,y_pred1)
print("SVC MODEL")

from sklearn import svm

svm=svm.SVC()
svm.fit(X_train,y_train)
y_pred2 =svm.predict(X_test)
accuracy_score(y_test,y_pred2)
precision_score(y_test,y_pred2)
recall_score(y_test,y_pred2)
f1_score(y_test,y_pred2)

from sklearn.neighbors import KNeighborsClassifier

knn= KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred3=knn.predict(X_test)
accuracy_score(y_test,y_pred3)
precision_score(y_test,y_pred3)
recall_score(y_test,y_pred3)
f1_score(y_test,y_pred3)

from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred4=dt.predict(X_test)
accuracy_score(y_test,y_pred4)
precision_score(y_test,y_pred4)
recall_score(y_test,y_pred4)
f1_score(y_test,y_pred4)

 from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred5=rf.predict(X_test)
accuracy_score(y_test,y_pred5)
precision_score(y_test,y_pred5)
recall_score(y_test,y_pred5)
f1_score(y_test,y_pred5)

from sklearn.ensemble import GradientBoostingClassifier

gb=GradientBoostingClassifier()
gb.fit(X_train,y_train)
y_pred6=gb.predict(X_test)
accuracy_score(y_test,y_pred6)
precision_score(y_test,y_pred6)
recall_score(y_test,y_pred6)
f1_score(y_test,y_pred6)

final_data=pd.DataFrame({'Models':['LR','SVC','KNN','DT','RF','GB'],'ACC':[accuracy_score(y_test,y_pred1),
                                                                           accuracy_score(y_test,y_pred2),
                                                                           accuracy_score(y_test,y_pred3),
                                                                           accuracy_score(y_test,y_pred4),
                                                                           accuracy_score(y_test,y_pred5),
                                                                           accuracy_score(y_test,y_pred6)]})
final_data

import seaborn as sns

sns.barplot(final_data['Models'],final_data['ACC'])

final_data=pd.DataFrame({'Models':['LR','SVC','KNN','DT','RF','GB'],'PRE':[precision_score(y_test,y_pred1),
                                                                           precision_score(y_test,y_pred2),
                                                                           precision_score(y_test,y_pred3),
                                                                           precision_score(y_test,y_pred4),
                                                                           precision_score(y_test,y_pred5),
                                                                           precision_score(y_test,y_pred6)]})
final_data

sns.barplot(final_data['Models'],final_data['PRE'])
final_data=pd.DataFrame({'Models':['LR','SVC','KNN','DT','RF','GB'],'RE':[recall_score(y_test,y_pred1),
                                                                           recall_score(y_test,y_pred2),
                                                                           recall_score(y_test,y_pred3),
                                                                           recall_score(y_test,y_pred4),
                                                                           recall_score(y_test,y_pred5),
                                                                           recall_score(y_test,y_pred6)]})
final_data

sns.barplot(final_data['Models'],final_data['RE'])
final_data=pd.DataFrame({'Models':['LR','SVC','KNN','DT','RF','GB'],'f1':[f1_score(y_test,y_pred1),
                                                                           f1_score(y_test,y_pred2),
                                                                           f1_score(y_test,y_pred3),
                                                                           f1_score(y_test,y_pred4),
                                                                           f1_score(y_test,y_pred5),
                                                                           f1_score(y_test,y_pred6)]})
final_data

sns.barplot(final_data['Models'],final_data['f1'])
X_res=sc.fit_transform(X_res)
rf.fit(X_res,y_res)

import joblib

joblib.dump(rf,'churn_predict_model')
model=joblib.load('churn_predict_model')
data.columns
model.predict ([[601,42,1,98495.72,1,1,0,40014.76,0,0,1]])

from tkinter import *
from sklearn.preprocessing import StandardScaler
import joblib
def show_entry_fields():
    p1=int(e1.get())
    p2=int(e2.get())
    p3=int(e3.get())
    p4=float(e4.get())
    p5=int(e5.get())
    p6=int(e6.get())
    p7=int(e7.get())
    p8=float(e8.get())
    p9=int(e9.get())
    if p9== 1:
        Geography_Bhaktapur=1
        Geography_Lalitpur=0
        Geography_Kathmandu=0
    elif p9== 2:
        Geography_Bhaktapur=0
        Geography_Lalitpur=1
        Geography_Kathmandu=0
    elif p9== 3:
        Geography_Bhaktapur=0
        Geography_Lalitpur=0
        Geography_Kathmandu=1
    
    p10=int(e10.get())
    model = joblib.load('churn_predict_model')
    result=model.predict(sc.transform([[p1,p2,p3,p4,p5,p6,p7,p8,Geography_Kathmandu,Geography_Lalitpur,p10]]))
    
    
    if result== 0:
        Label(master, text="Not churning").grid(row=31)
       
    else:
        Label(master, text="Churning").grid(row=31)
            
master= Tk()
master.title("Bank Customers Churn Predicition Using Machine Learning")

label = Label(master, text ="Customers Churn Prediciton using ML", bg="black", fg = "white"). \
                                                                        grid(row=0, columnspan=2)



Label(master, text="CreditScore").grid(row=1)
Label(master, text="Age").grid(row=2)
Label(master, text="Tenure").grid(row=3)
Label(master, text="Balance").grid(row=4)
Label(master, text="NumOfProducts").grid(row=5)
Label(master, text="HasCrCard").grid(row=6)
Label(master, text="IsActiveMember").grid(row=7)
Label(master, text="EstimatedSalary").grid(row=8)
Label(master, text="Geography").grid(row=9)
Label(master, text="Gender").grid(row=10)




e1= Entry(master)
e2= Entry(master)
e3= Entry(master)
e4= Entry(master)
e5= Entry(master)
e6= Entry(master)
e7= Entry(master)
e8= Entry(master)
e9= Entry(master)
e10= Entry(master)


e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)            
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)
e7.grid(row=7, column=1)
e8.grid(row=8, column=1)
e9.grid(row=9, column=1)
e10.grid(row=10, column=1)

Button(master, text='Predict', command=show_entry_fields).grid()

master.mainloop()

show_entry_fields()