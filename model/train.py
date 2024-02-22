import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

def model_train(train_df: pd.DataFrame) -> RandomForestClassifier:
    cat_cols = [
    'Sex',
    'Embarked'
    ]
    num_cols = [
    'Pclass',
    'Age',
    'SibSp',
    'Parch',
    'Fare'
    ]
    target_col = 'Survived'
    
    X_train = train_df.loc[:, cat_cols + num_cols].fillna({'Age':0})
    X_train_h = pd.get_dummies(X_train, columns=cat_cols) 
    y_train = train_df[target_col]

    model = RandomForestClassifier(max_depth=15, random_state=42)
    model.fit(X_train_h, y_train)    
    return model

if __name__ == '__main__':
    current_file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(current_file_dir, 'data', 'train.csv')
    model_path = os.path.join(current_file_dir, 'data', 'trained_model.sav')
    
    train_df = pd.read_csv(data_path)
    model = model_train(train_df)
    joblib.dump(model, model_path)


