import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,r2_score,mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler

def predict_data():
    df=pd.read_csv('world_trade_growth_clean.csv')

    # this is the little description of the data 
    print("this is the little description of the data ")
    print(f"data types : {df.dtypes}")
    print(f"Columns : {df.columns}")
    print(f"Descriptions : {df.describe()}")


    # divide the numerical and the categorical columns 

    catagorical_col=['country_code','region']

    numerical_col=['year']

    # this is the train test split model and the like of the main thing of the model

    x=df.drop(['trade_growth_rate','period','income_group','iso2_code','country_name'], axis=1)
    y=df['trade_growth_rate']

    x_train,x_test,y_train,y_test=train_test_split(
        x,y,
        test_size=0.2,
        random_state=52
    )

    print(f"X_train {x_train.shape}")
    print(f"X_test {x_test.shape}")

    # now the conversion of the like of the 

    x_train=pd.get_dummies(x_train, columns=catagorical_col, drop_first=True)
    x_test=pd.get_dummies(x_test, columns=catagorical_col, drop_first=True)

    x_test=x_test.reindex(columns=x_train.columns, fill_value=0)


    # this is the step of the scaling feature 

    le=StandardScaler()
    available_col=[col for col in numerical_col if col in x_train.columns]

    x_train[available_col]=le.fit_transform(x_train[available_col])
    x_test[available_col]=le.transform(x_test[available_col])

    model=RandomForestRegressor()

    model.fit(x_train, y_train)

    y_pred=model.predict(x_test)

    mae=mean_absolute_error(y_test, y_pred)
    r2=r2_score(y_test, y_pred)

    print("\n MODEL PERFORMANCE:")
    print(f"   r² Score: {r2:.3f} (1.0 = perfect prediction)")
    print(f"   Mean Absolute Error: {mae:,.2f}")
    print(f"   Average prediction is off by: {mae:,.2f}")

    


predict_data()
