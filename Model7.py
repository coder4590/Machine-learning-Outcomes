import pandas as pd 
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error , accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def analyze_data():
    df=pd.read_csv('Grammy_Awards_Winners_20260208_055452.csv')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50) 

    # this is the analysis of the whole data and the liike of the this thing is good and the like of the good

    print(" this is the classification report and the all of the thing of the like: ")
    print(f"Visual trends: {df.plot}")
    print(f"Day by Day Change {df.columns}")

    print("Now this is the like of the distrubation pattern: ")
    print(f"Distrubation shape: {df.hist()}")
    print(f"Frequency of the values {df.value_counts()}")
    print(f"Summary describation state {df.describe()}")

    # this is the analysis of the yearly pattern like how  can do this to this like of how 
    # now the other thing is to find the like of the yearly trend which help us to find the like of the the yeraly 

    df['Collection_Date']=pd.to_datetime(df['Collection_Date'])
    winner_by_decade=df.groupby('Decade')['Winner'].value_counts().unstack().fillna(0)
    winner_by_era=df.groupby('Era')['Winner'].value_counts().unstack().fillna(0)
    winner_by_year=df.groupby('Year')['Winner'].value_counts().unstack().fillna(0)
    winner_by_category=df.groupby('Category')['Winner'].value_counts().unstack().fillna(0)
    # now also we can find the trends of the monthly winner and the like of the decade wise winner you using hte like of the pandas and the keyword of the group by 

    decade_pattern=df.groupby(['Decade','Category']).size().unstack().fillna(0)
    award_group_trend=df.groupby(['Year','Award_Group']).size().unstack().fillna(0)
    

    print(winner_by_category)

def analyze_visually():
    df=pd.read_csv('Grammy_Awards_Winners_20260208_055452.csv')

    # like now we will like of the visualize the all of the trens which like help us to make the understanding of the pattern and the like of the understand the most of the pattern which help us in the like of the data analytic 
    fig =plt.figure(figsize=(20,15))
    # like we are saying that the visual trend is acutally is the award by the winner so like ye 
    plt.subplot(3,3,1)
    df['Year'].value_counts().sort_index().plot(kind='line', marker='o')
    plt.title('visual trends by the year ')
    plt.xlabel('Year')
    plt.ylabel('Count')

    # now this is the visualization of the like of the Day by Day change 

    plt.subplot(3,3,2)
    df['Category'].value_counts().plot(kind='bar')
    plt.title('Day by Day change ')
    plt.xticks(rotation=45)

    # Now the third thing is the like of the freqency distribution and the like of the this is the way of thhe viusalize the frequency distribution 
    plt.subplot(3,3,3)
    df['Award_Group'].value_counts().plot(kind='bar')
    plt.title('Award Groups')   

    # now the next thing we find is the like of the the main thing in the like of the we find using the pandas we use the same pandas but in the different way 

    plt.subplot(3,3,4)
    winner_by_decade=df.groupby('Decade')['Winner'].count()
    winner_by_decade.plot(kind='bar')
    plt.xticks(rotation=45)

    # Now the next thing is the like of the we find is the like of the 

    plt.subplot(3,3,5)
    winner_by_era=df.groupby('Era')['Winner'].count()
    winner_by_decade.plot(kind='bar')
    plt.title('Winners by Decade')
    plt.xticks(rotation=45)
    
    # now the next thing is that we find the like of the all of the other trends which will remain isthe like of the detail of the and the like and the main of the event and the like of the event and the this is not the nextthing we need to learn nowand the next thing we find the 

    plt.tight_layout()
    plt.show()

    # Now this is how we will like of the who all of the data visualiztion we have done till nwo and this is the most of the good and the important step in this like of the step 

def Predict_data():
    df=pd.read_csv('Grammy_Awards_Winners_20260208_055452.csv')

    # now the first thing in the like of the prediction model is the like of the like creating the categroy of the column of the numriacal and the like of the other also 

    numrical_col=['Ceremony_Number','Total_Wins',
                  'Category_Total_Winners']
    categorical_col=['Category','Award_Group','Status','Data_Source']

    # now the next thing is to decide which column is perfect to drop or not and also like of the which column like and which column we are the predicting one these all of thing which we have to find now 

    x=df.drop(['Winner','Collection_Date','Year','Artist','Era','Decade'], axis=1)
    y=df['Year']

    x_train,x_test,y_train,y_test=train_test_split(
        x,y,
        test_size=0.3,
        random_state=42
    )

    print(f"X_train {x_train.shape}")
    print(f"X_test {x_test.shape}")

    # now the next thing is the thing which is the most common thing and this thing is the converting of the categorical column into the like of the numrical column 

    # and this is how we use the like of the one hot encoder 

    x_train=pd.get_dummies(x_train, columns=categorical_col, drop_first=True)
    x_test=pd.get_dummies(x_test, columns=categorical_col, drop_first=True)

    x_test=x_test.reindex(columns=x_train.columns, fill_value=0)


    # the next thingis the like of the scaling the numrical feautre and most of the thing is that this is hte like of the not so much importance like but it give lot ofthe effect on the model and like train the model well without it like the model will learn but not so better than the liek of using it now 

    available_col=[col for col in numrical_col if col in x_train.columns]

    scale=StandardScaler()
    x_train[available_col]=scale.fit_transform(x_train[available_col])
    x_test[available_col]=scale.transform(x_test[available_col])

    # Now the other thing is to decide which model is best for the prediction and i think the classfier is the bes for the prediciton 

    model=RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(x_train, y_train)

    # now the last thing is the like of the training and the prediction 

    y_pred=model.predict(x_test)

    r2=r2_score(y_test, y_pred)
    mae=mean_absolute_error(y_test, y_pred)

    print("\n MODEL PERFORMANCE:")
    print(f"   r² Score: {r2:.3f} (1.0 = perfect prediction)")
    print(f"   Mean Absolute Error: {mae:,.2f}")
    print(f"   Average prediction is off by: {mae:,.2f}")




Predict_data()