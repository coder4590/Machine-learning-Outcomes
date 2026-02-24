import pandas as pd 
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error , accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def predict_data():
    df=pd.read_csv('Student_placement_skills_2025.csv')

   # Now the first is to categorize teh data into the required formate and the other thing also 

    categorical_col=['Gender','Degree']
    numerical_col=['Age','Internships_Count','Projects_Count',
                     'Certifications_Count','Technical_Skills_Score_100',
                     'Communication_Skills_Score_100','Aptitude_Test_Score_100','Salary_Offered_USD','CGPA']
    

    # now the next thing is to drop the liike of the all of column and the prediciton column also 

    x=df.drop(['Placement_Offer','Student_ID'], axis=1)
    y=df['Placement_Offer']

    x_train,x_test,y_train,y_test=train_test_split(
        x,y,
        test_size=0.3,
        random_state=43
    )

    print(f"test Data: {x_test.shape}")
    print(f"training data: {x_train.shape}")

    # now the nest thing is the converion of the data 

    x_train=pd.get_dummies(x_train, columns=categorical_col, drop_first=True)
    x_test=pd.get_dummies(x_test, columns=categorical_col, drop_first=True)

    x_test=x_test.reindex(columns=x_train.columns, fill_value=0)


    # now the next step is thel ike of the scaling 

    scaler=StandardScaler()

    available_col=[col for col in numerical_col if col in x_train.columns]

    x_train[available_col]=scaler.fit_transform(x_train[available_col])
    x_test[available_col]=scaler.transform(x_test[available_col])

    model=RandomForestClassifier(n_estimators=100, random_state=43)

    model.fit(x_train, y_train)

    y_pred=model.predict(x_test)
    accuracy=accuracy_score(y_test, y_pred)

    print(f"accuracy: {accuracy:.3f}")
    print(f"Correct_prediction: {accuracy*100:.1f}")

    print("Classification Report")
    print(classification_report(y_test,y_pred))

    print("\nCONFUSION MATRIX:")
    print(confusion_matrix(y_test, y_pred))


    importance = pd.DataFrame({
    'feature': x_train.columns,
    'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)


    print(importance.head(10))


    plt.figure(figsize=(10,6))
    sns.barplot(data=importance.head(10), x='importance', y='feature')
    plt.title('Top 10 Most Important Features')
    plt.show()


   

predict_data()