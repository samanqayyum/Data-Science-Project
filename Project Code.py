
"""
Created on Sat Jun 15 22:42:57 2024

@author: Saman Qayyum
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import warnings
warnings.filterwarnings('ignore')

sd = '2014-01-01'
ed = '2024-12-31'
data = yf.download("MSFT" ,start=sd, end=ed)
data.to_csv("MSFT.csv")


df = pd.read_csv("MSFT.csv")


new_df = df[['Date', 'Adj Close','Volume']]
new_df['Date'] = pd.to_datetime(new_df['Date'])


def calculate_sma(ds, window):
    ds["SMA"]= ds['Adj Close'].rolling(window=window).mean()
    return ds

def calculate_bollinger_bands(ds, window, num_std_dev):
    ds['Bollinger Middle Band'] = ds['Adj Close'].rolling(window=window).mean()
    rolling_std = ds['Adj Close'].rolling(window=window).std()
    ds['Bollinger Upper Band'] = ds['Bollinger Middle Band'] + (rolling_std * num_std_dev)
    ds['Bollinger Lower Band'] = ds['Bollinger Middle Band'] - (rolling_std * num_std_dev)
    return ds


def calculate_obv(ds):

    obv = [0]
    
    for i in range(1, len(ds)):
        if ds['Adj Close'][i] > ds['Adj Close'][i - 1]:
            obv.append(obv[-1] + ds['Volume'][i])
        elif ds['Adj Close'][i] < ds['Adj Close'][i - 1]:
            obv.append(obv[-1] - ds['Volume'][i])
        else:
            obv.append(obv[-1])
    
    ds['OBV'] = obv
    return ds

def calculate_signals(ds):
    ds['Signal'] = 'Neutral'
    for i in range(1, len(ds)):
        
        if ds['Adj Close'].iloc[i] < ds['Bollinger Lower Band'].iloc[i]:
            ds.at[i, 'SignalBB'] = 'Buy'
        elif ds['Adj Close'].iloc[i] > ds['Bollinger Upper Band'].iloc[i]:
            ds.at[i, 'SignalBB'] = 'Sell'
        else:
           ds.at[i, 'SignalBB'] = 'Neutral'

        if  i==1:
            ds.at[i, 'SignalOBV'] = 'Neutral'
        if  ds['OBV'].iloc[i] > ds['OBV'].iloc[i - 1]:
            ds.at[i, 'SignalOBV'] = 'Buy'
        elif ds['OBV'].iloc[i] < ds['OBV'].iloc[i - 1]:
            ds.at[i, 'SignalOBV'] = 'Sell'
        else:
            ds.at[i, 'SignalOBV'] = 'Neutral'

        if ds['Adj Close'].iloc[i] < ds['SMA'].iloc[i]:
            ds.at[i, 'SignalSMA'] = 'Buy'
        elif ds['Adj Close'].iloc[i] > ds['SMA'].iloc[i]:
            ds.at[i, 'SignalSMA'] = 'Sell'
        else:
            ds.at[i, 'SignalSMA'] = 'Neutral'
    
    return ds

def signal_map(df):
    map = {'Buy': 1, 'Sell': -1, 'Neutral': 0}
    df['SignalSMA_Encoded'] = df['SignalSMA'].map(map)
    df['SignalOBV_Encoded'] = df['SignalOBV'].map(map)
    df['SignalBB_Encoded'] = df['SignalBB'].map(map)
    return df

new_df = calculate_sma(new_df,56)
new_df = calculate_bollinger_bands(new_df,window=3, num_std_dev=2)
new_df = calculate_obv(new_df)
new_df.dropna(inplace=True)
new_df = calculate_signals(new_df)
new_df = signal_map(new_df)
new_df.dropna(inplace=True)

def logistic_regression(df):
    # Create features and target
    features = df[['Adj Close', 'SMA']]
    target = df["SignalSMA_Encoded"]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model, scaler

print(new_df)

model, scaler = logistic_regression(new_df)