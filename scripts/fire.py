#!/usr/bin/env python
# coding: utf-8

##Data Manipulation
import pandas as pd
import numpy as np
from pandas import DataFrame,Series

#Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path

##Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,ElasticNet,Lasso,Ridge,LassoCV,RidgeCV,ElasticNetCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler,OneHotEncoder,OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error,root_mean_squared_error,r2_score
from sklearn.compose import ColumnTransformer
import pickle




# Define project root paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Go up from /notebook/
NOTEBOOK_DIR = PROJECT_ROOT / "notebook"
SAVE_MODELS_DIR = PROJECT_ROOT / "save_models"

# Create directories if they don't exist
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
SAVE_MODELS_DIR.mkdir(parents=True, exist_ok=True)




# ## Data extraction and Data Cleaning

def data_convert_clean(data_location,folder_name="notebook"):
    # Define numeric columns
    num_cols1 = ['day', 'month', 'year', 'temperature', 'rh', 'ws']
    num_cols2 = ['rain', 'ffmc', 'dmc', 'dc', 'isi', 'bui', 'fwi']

    # Read data
    df = pd.read_csv(data_location, header=1)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    # Find rows where all but the first column are NaN (separator rows)
    del_row = df[df.iloc[:, 1:].isnull().all(axis=1)].index

    split_point = None
    if not del_row.empty:
        split_point = del_row[0]   # first separator row index
        df = df.drop(index=del_row).reset_index(drop=True)
        df = df.drop(index=del_row).reset_index(drop=True)

    # Convert integer-like cols
    df[num_cols1] = df[num_cols1].astype('Int64')

    # Find invalid entries in numeric columns
    mask = ~df[num_cols2].applymap(
        lambda x: str(x).replace('.', '', 1).replace('-', '', 1).isdigit()
    )
    bad_idx = df[mask.any(axis=1)].index.tolist()

    for i in bad_idx:
        wrong_val = df.iloc[i, -2]  # second-to-last col (fwi)
        # Convert if numeric, else set NaN
        df.iloc[i, -2] = (
            float(wrong_val) if str(wrong_val).replace('.', '', 1).isdigit() else np.nan
        )
        # Fill last column (classes) if missing
        if pd.isna(df.iloc[i, -1]):
            df.iloc[i, -1] = df.iloc[:, -1].mode()[0]

    # Final conversion of numeric cols
    df[num_cols2] = df[num_cols2].replace(r'\s+', '', regex=True).astype(float)

    # Region assignment (dynamic)
    if split_point is not None:
        df.loc[:split_point-1, 'region'] = 'Bejaia'
        df.loc[split_point:, 'region'] = 'Sidi-Bel Abbes'

    # Clean class labels
    df['classes'] = df['classes'].str.strip()

    save_path=NOTEBOOK_DIR/"Clean_Fire_data.csv"
    df.to_csv(save_path,index=False)

    return df,save_path

# ========== Step 2: Read Cleaned Data ==========

def read_data():
    data_path=NOTEBOOK_DIR
    if not data_path.exists():
        raise FileNotFoundError(f'Folder is empty please check{str(data_path)}')

    files=[file for file in data_path.glob("*.csv")]
    if not files:
        raise ValueError(f'Folder is empty please check{str(data_path)}')

    raw_data=[pd.read_csv(file) for file in files]
    df=pd.concat(raw_data,ignore_index=False)
    return df

# ========== Step 3: Split Data ==========
def split_data(df,target):
    unused_cols=['day','month','year']
    df=df.drop(columns=unused_cols)

    df_train,df_tmp=train_test_split(df,test_size=0.4,random_state=42)
    df_valid,df_test=train_test_split(df_tmp,test_size=0.5,random_state=42)

    X_train=df_train.drop(columns=['classes',target])
    y_train=df_train[target]

    X_valid=df_valid.drop(columns=['classes',target])
    y_valid=df_valid[target]

    X_test=df_test.drop(columns=['classes',target])
    y_test=df_test[target]

    return X_train,y_train,X_valid,y_valid,X_test,y_test


# ========== Step 4: Save Splits==========
def save_split(X_train,y_train,X_valid,y_valid,X_test,y_test):

    train= pd.concat([X_train, y_train], axis=1)
    valid= pd.concat([X_valid, y_valid], axis=1)
    test= pd.concat([X_test, y_test], axis=1)

    ##train=np.concatenate((X_train, y_train.to_numpy().reshape(-1,1)), axis=1)
    ##valid=np.concatenate((X_valid, y_valid.to_numpy().reshape(-1, 1)), axis=1)
    ##test=np.concatenate((X_test, y_test.to_numpy().reshape(-1, 1)), axis=1)

    train_path=NOTEBOOK_DIR /"train.csv"
    test_path=NOTEBOOK_DIR /"test.csv"
    valid_path=NOTEBOOK_DIR /"valid.csv"

    
    train.to_csv(train_path,index=False)
    valid.to_csv(valid_path,index=False)
    test.to_csv(test_path,index=False)
    
    #pd.DataFrame(train).to_csv(train_path,index=False)
    #pd.DataFrame(valid).to_csv(valid_path,index=False)
    #pd.DataFrame(test).to_csv(test_path,index=False)


# ========== Step 5: Feature Engineering ==========
def feature_eng(X_train,y_train,X_valid,y_valid,X_test,y_test):

    num_cols=X_train.select_dtypes(include='number').columns.to_list()
    cat_cols=['region']

    num_trans=make_pipeline(SimpleImputer(strategy='mean'),StandardScaler())
    cat_trans=make_pipeline(SimpleImputer(strategy='most_frequent'),OrdinalEncoder())

    feature_transform=ColumnTransformer(
        transformers=[
            ('num',num_trans,num_cols),
            ('cat',cat_trans,cat_cols)
        ]
    )

    X_train=feature_transform.fit_transform(X_train)
    X_valid=feature_transform.transform(X_valid)
    X_test=feature_transform.transform(X_test)

    y_train=y_train.fillna(y_train.mean())
    y_valid=y_valid.fillna(y_valid.mean())
    y_test=y_test.fillna(y_test.mean())

    transformer_path = SAVE_MODELS_DIR / "feature_transformer.pkl"
    with open(transformer_path,"wb") as pkfile:
        pickle.dump(feature_transform,pkfile)

    return X_train,y_train,X_valid,y_valid,X_test,y_test

# ========== Step 6: Train Model ==========
def train_model(X_train,y_train,X_valid,y_valid):
    y_train = np.ravel(y_train)
    y_valid = np.ravel(y_valid)

    model=RidgeCV(cv=5)
    model.fit(X_train,y_train)

    pred=model.predict(X_valid)
    mae=mean_absolute_error(y_valid,pred)
    r2=r2_score(y_valid,pred)

    print('mean absolute error',mae)
    print('r2 score',r2)

    model_path = SAVE_MODELS_DIR / "model.pkl"

    with open(model_path,"wb") as pkfile:
        pickle.dump(model,pkfile)

    return model

# ========== Step 6: Run Pipeline ==========
def run_pipeline(raw_csv):
    # Clean & Save
    print("[INFO] Cleaning data...")
    df, _ = data_convert_clean(raw_csv)

    # Read Cleaned
    print("[INFO] Reading cleaned data...")
    df = read_data()

    # Split
    print("[INFO] Splitting data...")
    X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(df, 'fwi')

    # Save splits
    print("[INFO] Saving data splits...")
    save_split(X_train, y_train, X_valid, y_valid, X_test, y_test)

    # Feature Eng
    print("[INFO] Feature engineering...")
    X_train, y_train, X_valid, y_valid, X_test, y_test = feature_eng(
        X_train, y_train, X_valid, y_valid, X_test, y_test
    )

    # Train
    print("[INFO] Training model...")
    model=train_model(X_train, y_train, X_valid, y_valid)

    print("[INFO] Pipeline completed successfully!")
    print(f"Cleaned data and splits saved to: {NOTEBOOK_DIR}")
    print(f"Model and transformer saved to: {SAVE_MODELS_DIR}")


# ========== Main ==========
if __name__ == "__main__":
    RAW_DATA = r"C:\Users\HP\Downloads\Ridge+Lassso+Elastic+Regression+Practicals\Ridge Lassso Elastic Regression Practicals\Algerian_forest_fires_dataset_UPDATE.csv"
    run_pipeline(RAW_DATA)

