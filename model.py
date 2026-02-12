import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error



def analyse_the_data():
    df=pd.read_csv('Housing2.csv')

    price_summary=df['price'].describe()

    # Now we will find the interquantile of the prices and the other thing 

    q1=df['price'].quantile(0.25)
    q3=df['price'].quantile(0.75)

    # Now we will find the like of the inter quantile range which show us how much is the data is the bizare in the real world

    iqr=q3-q1
    # The next thing we will find is the outlier fence and this is like of the important one for this like of the method 
    threshold=q3 + 1.5 * iqr

    print(price_summary)
    print(iqr)

    outliers=df[df['price']>threshold]

    investigation_df=outliers.copy()

    print(investigation_df[['price','bedrooms','bathrooms','stories','hotwaterheating','airconditioning','guestroom','parking','prefarea','furnishingstatus']])

def luxury_house():
    df=pd.read_csv('Housing2.csv')

    # now i will define the luxury in this data st 

    luxury=df[(df['price']>df['price'].quantile(0.95))&
              (df['area']>df['area'].median())&
              (df['airconditioning']=='yes')&
              (df['furnishingstatus']=='funished')&
              (df['parking']>=2)]
    # Luxury Type 1: The Mansion
    mansion = df[
    (df['price'] > df['price'].quantile(0.90)) &
    (df['area'] > df['area'].quantile(0.75)) &
    (df['bedrooms'] >= 4) &
    (df['bathrooms'] >= 3)
]

# Luxury Type 2: The Premium Location Home
    location_luxury = df[
    (df['price'] > df['price'].quantile(0.90)) &
    (df['prefarea'] == 'yes') &
    (df['area'] < df['area'].median())  # Small but expensive!
]

# Luxury Type 3: The Fully Loaded Home
    feature_luxury = df[
    (df['price'] > df['price'].quantile(0.90)) &
    (df['airconditioning'] == 'yes') &
    (df['furnishingstatus'] == 'furnished') &
    (df['hotwaterheating'] == 'yes')
]

# Luxury Type 4: The Entertainer's Home
    entertainer = df[
    (df['price'] > df['price'].quantile(0.90)) &
    (df['guestroom'] == 'yes') &
    (df['parking'] >= 2) &
    (df['stories'] >= 2)
]

    print(f"Mansions: {len(mansion)}")
    print(f"Location Luxury: {len(location_luxury)}")
    print(f"Feature Luxury: {len(feature_luxury)}")
    print(f"Entertainer's Home: {len(entertainer)}")
    print(f"My definition finds {len(luxury)} luxury houses")
    
    # this is the like of the my own trade mark defining the luxury but hte data which i have is the like of the denying it and its really like i am frustrated this nnow and this sdkl

    






    


