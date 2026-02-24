import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def analyze_data():
    df=pd.read_csv('ecomerce_revnue_growth.csv')

    # Now we know that like of the we need to doo all of the data cleaning and the data analysis ofthe data which is known as the data analyze and this is the thing we need to do in like of the ourdata set and this is the n

    print(f"The datatype of the all of the columns is :{df.dtypes}")
    print(f"The detail of the like of all of the column is : {df.describe()}")
    print(f"columns is like of the : {list(df.columns)}")

    # Now we have to find the relation ship and like of the all of the column which is effecting each other and the other thing also 

    df['order_date'] = pd.to_datetime(df['order_date'])
    df['month'] = df['order_date'].dt.month
    df['day_of_week'] = df['order_date'].dt.day_name()
    df['week_number'] = df['order_date'].dt.isocalendar().week

    # First of all we will calcualte the montly revnue which is the use of the pandas 

    monthly_revenue=df.groupby('month')['revenue'].sum().reset_index()
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.barplot(data=monthly_revenue, x='month', y='revenue')
    plt.title('Revenue by Month')
    plt.xlabel('Month')
    plt.ylabel('Total Revenue')
    
    # Now we will create the monthly trendo of the like of the days like which day get the highest sale in the year

    highes_sale=df.groupby('day_of_week')['revenue'].mean().reset_index()
    day_order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    highes_sale['day_of_week']=pd.Categorical(highes_sale['day_of_week'], categories=day_order, ordered=True)
    highes_sale=highes_sale.sort_values('day_of_week')

    plt.subplot(1,2,2)
    sns.barplot(data=highes_sale, x='day_of_week', y='revenue')
    plt.title('Average revenue by day of the week')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Now the other thing we will calculate the is the time casting and all of the other concept which is like of the backbone of the ecommerce 

    
    

def clean_model():
    df=pd.read_csv('ecomerce_revnue_growth.csv')

    # this is to clean the data 
    df = df.drop_duplicates(subset=['order_id'], keep='first')
    
    df=df[df['age']>13]

    df.to_csv("ecomerce_revnue_growth_cleaned.csv")


def predict_data():
    df=pd.read_csv('ecomerce_revnue_growth.csv')

    # First we will like preprocess the data 
    # first we will like of the categorize the data

    catogrical_column=['state','zone','category','brand_type','customer_gender','sales_event'
                       ,'competition_intensity','inventory_pressure']
    
    numerical_column=['customer_age']


    # Now the next step is the like of the test train and the split 
    
    x=df.drop(['units_sold','order_id','order_date','base_price'
               ,'discount_percent','final_price'], axis=1)
    y=df['units_sold']

    x_train,x_test,y_train,y_test=train_test_split(
        x,y,
        test_size=0.2,
        random_state=42
    )

    print(f"X_train {x_train.shape}")
    print(f"X_test {x_test.shape}")

    # Now the next step is the like of the converting the data using the onhotencoding now we don't use the like of the onhot encoder with the piple line we will leanr in the future 

    x_train=pd.get_dummies(x_train, columns=catogrical_column, drop_first=True)
    x_test=pd.get_dummies(x_test, columns=catogrical_column, drop_first=True)

    x_test=x_test.reindex(columns=x_train.columns, fill_value=0 )

    # Now the next step is the like of the stander scaler which is used to scale the calue of the all of the numrical category is this is the like of the step is which is not like of super important but for 
    available_col=[ col for col in numerical_column if col in x_train.columns]
    scale=StandardScaler()
    x_train[available_col]=scale.fit_transform(x_train[available_col])
    x_test[available_col]=scale.transform(x_test[available_col])

    # now the next step is the like of the verification of the model which is buil 

    model=RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(x_train, y_train)

    # now the next step is the like of the prediction the column and this is the like of the 
    y_pred=model.predict(x_test)

    mae=mean_absolute_error(y_test, y_pred)
    r2=r2_score(y_test, y_pred)

    print("\n MODEL PERFORMANCE:")
    print(f"   r² Score: {r2:.3f} (1.0 = perfect prediction)")
    print(f"   Mean Absolute Error: {mae:,.2f}")
    print(f"   Average prediction is off by: {mae:,.2f}")



predict_data()


