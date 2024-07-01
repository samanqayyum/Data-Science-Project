
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


sd = '2014-01-01'
ed = '2024-12-31'
data = yf.download("MSFT" ,start=sd, end=ed)
data.to_csv("MSFT.csv")


df = pd.read_csv("MSFT.csv")


new_df = df[['Date', 'Adj Close','Volume']]
new_df['Date'] = pd.to_datetime(new_df['Date'])
def calculate_sma(ds, window):
    ds["SMA"]= ds['Adj Close'].rolling(window=window).mean()

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
        if ds['Adj Close'][i] < ds['Bollinger Lower Band'][i] and ds['OBV'][i] > ds['OBV'][i - 1]:
            ds.at[i, 'Signal'] = 'Buy'
        elif ds['Adj Close'][i] > ds['Bollinger Upper Band'][i] and ds['OBV'][i] < ds['OBV'][i - 1]:
            ds.at[i, 'Signal'] = 'Sell'
        else:
            ds.at[i, 'Signal'] = 'Neutral'
    
    return ds


calculate_sma(new_df,56)
calculate_bollinger_bands(new_df,window=3, num_std_dev=2)
calculate_obv(new_df)
calculate_signals(new_df)
print(new_df)