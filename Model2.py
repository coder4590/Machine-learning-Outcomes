import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def data_analysis():
    df=pd.read_csv('background_noise_focus_dataset.csv')

    # this is to clean the data 

    # first remove the impossible 
    df = df.drop_duplicates(subset=['participant_id'], keep='first')

    df=df[  (df['noise_volume_level'] >=1)&
                 (df['focus_duration_minutes']>=1)&
                 (df['perceived_focus_score']>=1)&
                 (df['task_completion_quality']>=1)&
                 (df['mental_fatigue_after_task']>=0)]
    
    df.to_csv('background_noise_focus_dataset_cleaned.csv', index=False)

    print("The original file is updated successfuly and without any other error ")


def predict_data():
    df=pd.read_csv('background_noise_focus_dataset_cleaned.csv')


    def group_score(score):
        if score <=3:
            return 'low'
        elif score <=7:
            return 'meduim'
        else:
            return 'high'
        

    
    df['focus_level']=df['perceived_focus_score'].apply(group_score)
    # Now the next task is the data preprocessing 

    # Now first is the understand the data 

    print("Data types of the all of the columns is as : ")
    print(f"Columns {df.columns}")
    print(f"datatypes: {df.dtypes}")

    # Now the next step is to like of the change or the like of the numerical and the catogorical column

    numerical_col=['age','noise_volume_level','focus_duration_minutes',]
    
    categorical_col=['role','task_type',
                     'background_noise_type']
    
    # Now the third task is the train split and the like of the test the data

    x=df.drop(['participant_id','perceived_focus_score','task_completion_quality'
               ,'mental_fatigue_after_task','focus_level'], axis=1)
    y=df['focus_level']

    x_train, x_test, y_train, y_test=train_test_split(
        x,y,
        test_size=0.2,
        random_state=42
        )
        
    

    print(f"X_train {x_train.shape}")
    print(f"X_test {x_test.shape}")

    # Now we conver the data like of the data which is in the like of the string into the like of the numerical data 
    
    # Now for this method or for the converiosn of the this bizer data we use the like of the pandas and the 
    # the keyword is the like of the get dummies okay now tell

    x_train=pd.get_dummies(x_train, columns=categorical_col, drop_first=True)
    x_test=pd.get_dummies(x_test, columns=categorical_col, drop_first=True)

    # Now this is use to align the like of the test column with the like of the train column and this is very essential for this reason and for the specific reason 

    x_test=x_test.reindex(columns=x_train.columns, fill_value=0)

    # Now the last step is to scale the featuer and like learn the model how it is working and what is the true or not 


    scaler=StandardScaler()
    numerical_col_present=[col for col in numerical_col if col in x_train.columns]
    x_train[numerical_col_present]=scaler.fit_transform(x_train[numerical_col_present])
    x_test[numerical_col_present]=scaler.transform(x_test[numerical_col_present])


    model=RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(x_train, y_train)

    y_pred=model.predict(x_test)

    accuracy=accuracy_score(y_test,y_pred)
    print(f"accuracy: {accuracy:.3f}")
    print(f"Correct_prediction: {accuracy*100:.1f}")

    print("Classification Report")
    print(classification_report(y_test,y_pred))

    print("\nCONFUSION MATRIX:")
    print(confusion_matrix(y_test, y_pred))

    new_person = pd.DataFrame([{
        'age': 25,
        'role': 'Student',
        'task_type': 'Studying',
        'background_noise_type': 'Silence',
        'noise_volume_level': 1,
        'focus_duration_minutes': 90
    }])
    
    # 2. One-hot encode (no encoder saved, so use get_dummies)
    new_encoded = pd.get_dummies(new_person, 
                                 columns=['role', 'task_type', 'background_noise_type'],
                                 drop_first=True)
    
    # 3. CRITICAL - Align columns with training!
    new_encoded = new_encoded.reindex(columns=x_train.columns, fill_value=0)
    
    # 4. Scale numerical columns
    numerical_cols = ['age', 'noise_volume_level', 'focus_duration_minutes']
    new_encoded[numerical_cols] = scaler.transform(new_encoded[numerical_cols])
    
    # 5. Predict!
    prediction = model.predict(new_encoded)[0]
    
    print(prediction)



predict_data()




