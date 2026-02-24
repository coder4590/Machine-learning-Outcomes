import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

def clean_data():
    df=pd.read_csv('Titanic-Dataset.csv')

    df = df.drop_duplicates(subset=['PassengerId'], keep='first')

    df=df[(df['Age']>0)&
          (df['Fare']>0)]
    
    df.to_csv('Titanic-Dataset-cleaned.csv')


def predict_data():
    df=pd.read_csv('Titanic-Dataset.csv')

    # this is the little analysis of the data before the like of the preprocessingand like of the good practice and good for the future use 

    print("this is the little description of the data ")
    print(f"data types: {df.dtypes}")
    print(f"column: {df.columns}")
    print(f"Description: {df.describe()}")


    # The first step is the like of the separating of the column of the numerical and the categorical data 

    Numerical_col=['Age','SibSp',
                   'Parch','Fare','Pclass']
    
    Categorical_col=['Name','Sex','Ticket',
                     'Cabin','Embarked']
    
    

    # The next step is of the like of the train test and choosing which column is best to drop during the prediction 

    x=df.drop(['PassengerId','Survived',],axis=1)
    y=df['Survived']

    x_train,x_test,y_train,y_test=train_test_split(
        x,y,
        test_size=0.3,
        random_state=42
    )

    print(f"X_train {x_train.shape}")
    print(f"X_test {x_test.shape}")

    # Now the next thing is the like of the choosing like of the converting the catogorical column into thelike of the numrical column 

    # So now we use the onehot encoder techinique for this task 

    x_train=pd.get_dummies(x_train, columns=Categorical_col, drop_first=True)
    x_test=pd.get_dummies(x_test, columns=Categorical_col, drop_first=True)

    x_test=x_test.reindex(columns=x_train.columns, fill_value=0)

    # Now the next step is the scaling of the featuer and this is the like of the good thing to do now to doo all of the scaling of the feature and like of the good thing to do now 

    available_col=[col for col in Numerical_col if col in x_train.columns]

    scaler=StandardScaler()

    x_train[available_col]=scaler.fit_transform(x_train[available_col])
    x_test[available_col]=scaler.transform(x_test[available_col])

    model=RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(x_train,y_train)

    # Now the next thing is the prediciton using hte y prediciton 

    y_pred=model.predict(x_test)

    accuracy=accuracy_score(y_test, y_pred)

    print(f"accuracy: {accuracy:.3f}")
    print(f"Correct_prediction: {accuracy*100:.1f}")

    print("Classification Report")
    print(classification_report(y_test,y_pred))

    print("\nCONFUSION MATRIX:")
    print(confusion_matrix(y_test, y_pred))


predict_data()






