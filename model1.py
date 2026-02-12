import pandas as pd
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def data_cleaning():
    df=pd.read_csv('Housing.csv')

    print(df)



# First of the all i have to clean the data set which is given to me 
# second of the all i have to preprocess that data set which is given to me 
# third to the all i have to do the data analysis of the given data set which is given to me now 
# fourth of the all i have to the Build the model on the specific data set which is given to me now 

# Then the interviewer asked or used that data set to like of the predict the prices of the housed using the like of the house square fit value this is the most of the crucial part of this 

# 1. The first step is the cleaning of the data which is given to us and the 
# Cleaning of the data mean the removel of the rows and data which is not fit in the scope of our data set this is 
# This is the most of the crucial step in the cleaning of the data 

    df=df.dropna()

    df=df[df['price'] > 0 ]
    df=df[df['area'] > 0 ]
    df=df[df['bedrooms'] >= 1]
    df=df[df['bathrooms'] >= 1]
    df=df[df['stories'] >= 1]
    

    df.to_csv('housing.csv', index=False)
    print("The original file is updated successfuly and without any other error ")
    print(f"new shape {df.shape}")

    # from here the data preprocessing is started which is hte most of the crucial step in the machine learning 

    print(df.head())
    print(f"/columns {list(df.columns)}")
    print(f"Data types {df.dtypes}")    

    # Now we have to do the two main thing which is the most of the essential thing for the data proprecessing 

    #A. firt separate or find the all of the column which is like of the number int 
    #B. second separate all of the file which is like of the all of the like of the chart or integer 

    numeraic_col=['bedrooms', 'bathrooms','stories', 'area', 'parking']
    print(f"Numerical columns {numeraic_col}")

    catogory_col=['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
                  'prefarea', 'furnishingstatus']
    print(f"category columns: {catogory_col}")

    x=df.drop('price', axis=1)
    y=df['price']

    x_train, x_test, y_train, y_test=train_test_split(
        x,y,
        test_size=0.2,
        random_state=42
    )

    print(f"Training data: {x_train.shape}")
    print(f"Testing data : {x_test.shape}")

    # Now the next step is very important in which we have to handl the all of the other catogorical data which in the form of the yes or not 
    # becaues the machine learning libraries only understand the like of the 1 of the 0 and the nothing else so 
    # So its is very important to convert this data into the desired shape 

    encoder={}

    for column in ['mainroad', 'guestroom', 'basement', 
               'hotwaterheating', 'airconditioning', 'prefarea']:
        le=LabelEncoder()
        x_train[column]=le.fit_transform(x_train[column]) # this is like the fit on the train
        x_test[column]=le.transform(x_test[column]) # and this is like of the apply to test 
        encoder[column] = le 

    le=LabelEncoder()
    # this is for the like of the furnihsed status because he is neither yes nor no he hes is like yes or not 

    x_train['furnishingstatus']=le.fit_transform(x_train['furnishingstatus'])
    x_test['furnishingstatus']=le.transform(x_test['furnishingstatus'])
    encoder['furnishingstatus']=le

    print("After encoding the catogorical data ")
    print(x_train.head())

    scaler=StandardScaler()

    x_train[numeraic_col]=scaler.fit_transform(x_train[numeraic_col])

    x_test[numeraic_col]=scaler.transform(x_test[numeraic_col])

    model=LinearRegression()

    model.fit(x_train, y_train) 

    print ("Model leaning is compeleted and done it successfully")
    print(f"Model learned {len(model.coef_)} coffient ")
    print(f"Model intercept {model.intercept_:,.0f}")

    y_pred=model.predict(x_test)

    r2=r2_score(y_test, y_pred)
    mae=mean_absolute_error(y_test, y_pred)

        
    print("\n MODEL PERFORMANCE:")
    print(f"   r² Score: {r2:.3f} (1.0 = perfect prediction)")
    print(f"   Mean Absolute Error: ${mae:,.0f}")
    print(f"   Average prediction is off by: ${mae:,.0f}")

    # Your new house dictionary
    new_house = {
    'area': 3000,
    'bedrooms': 4,
    'bathrooms': 2,
    'stories': 2,
    'parking': 2,
    'mainroad': 'yes',
    'guestroom': 'no',
    'basement': 'yes',
    'hotwaterheating': 'no',
    'airconditioning': 'yes',
    'prefarea': 'yes',
    'furnishingstatus': 'furnished'
}

    new_house_df = pd.DataFrame([new_house])

# Encode binary columns
    for column in ['mainroad', 'guestroom', 'basement', 
                   'hotwaterheating', 'airconditioning', 'prefarea']:
        new_house_df[column] = encoder[column].transform(new_house_df[column])  # ← No fit()!
    
    new_house_df['furnishingstatus'] = encoder['furnishingstatus'].transform(new_house_df['furnishingstatus'])  # ← No fit()!

# Scale numerical columns
    new_house_df[numeraic_col] = scaler.transform(new_house_df[numeraic_col])

# CRITICAL FIX - Force SAME column order as training! ===
    new_house_df = new_house_df[x_train.columns]  

# Now predict
    price = model.predict(new_house_df)[0]  
    print(f"Price: ${price:,.0f}")











