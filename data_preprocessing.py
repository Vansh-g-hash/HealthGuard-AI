"""data_preprocessing.py
Load the Pima Indians Diabetes Dataset, do basic cleaning and feature preparation.
Expects `diabetes.csv` in the same folder.
"""
import pandas as pd
import numpy as np

def load_data(path='diabetes.csv'):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    # Replace zeros in certain columns with NaN (they indicate missing measurements)
    cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zero:
        df[col] = df[col].replace(0, pd.NA)
    # Fill missing values with median
    df = df.fillna(df.median())
    return df

def feature_target_split(df):
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    return X, y

if __name__ == '__main__':
    df = load_data()
    print('Raw shape:', df.shape)
    df = clean_data(df)
    X, y = feature_target_split(df)
    print('Features shape:', X.shape, 'Target shape:', y.shape)
