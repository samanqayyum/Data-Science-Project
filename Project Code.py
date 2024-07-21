
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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')


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
    #ds['Signal'] = 'Neutral'
    for i in range(0, len(ds)):
        
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
    df.to_csv("test.csv")
    return df

def check_profit(price, signal, start_cash):
    current_cash = start_cash
    current_stock = 0

    for i in range(len(price)):
        if signal.iloc[i] == 1 and current_cash > 0:
            # Buy stocks
            current_stock = current_cash / price.iloc[i]
            current_cash = 0
        elif signal.iloc[i] == -1 and current_stock > 0:
            # Sell stocks
            current_cash = current_stock * price.iloc[i]
            current_stock = 0
        #    print(signal.index[i], "Sell")
        #print(current_cash,signal.iloc[i])

    final_value = current_cash + current_stock * price.iloc[-1]
    profit = final_value - start_cash
    profit_percentage = (profit / start_cash) * 100
    print("start_cash",start_cash,'final_value',final_value,"profit",profit,"profit_percentage",profit_percentage)
    return profit, profit_percentage

def logistic_regression(df, feature, target):
    # Create features and target
   
    cols = ['Adj Close']
    for f in feature:
        cols.append(f)

    features = df[cols]
    target = df[target]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
  
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the logistic regression model
    print(y_train.value_counts())
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model, scaler

def random_forest(df, feature, target):

    cols = ['Adj Close']
    for f in feature:
        cols.append(f)

    features = df[cols]
    target = df[target]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)
    print(f"Random Forest Model Accuracy: {accuracy * 100:.2f}%")
    return model, scaler

def support_vector_machine(df, feature, target):
    cols = ['Adj Close']
    for f in feature:
        cols.append(f)

    features = df[cols]
    target = df[target]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel='linear')
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)
    print(f"Support Vector Machine Model Accuracy: {accuracy * 100:.2f}%")
    return model, scaler



def process_indicator(df_filter,indicators,signal,signal_encoded ):
    ind_size = len(indicators)
    modellr, scaler = logistic_regression(df_train,indicators,signal_encoded)
    modelrf, _ = random_forest(df_train,indicators,signal_encoded)
    modelsvm, _ = support_vector_machine(df_train,indicators,signal_encoded)



    plt.figure(figsize=(14, 7))
    plt.plot(df_filter['Date'], df_filter['Adj Close'], label='Adj Close', color='blue')
    if ind_size>0:
        plt.plot(df_filter['Date'], df_filter[indicators[0]], label=indicators[0], color='orange')
    if ind_size>1:
        plt.plot(df_filter['Date'], df_filter[indicators[1]], label=indicators[1], color='orange')
    
    df_buy_signals = df_filter[df_filter[signal] == 'Buy']
    df_sell_signals = df_filter[df_filter[signal] == 'Sell']
    plt.scatter(df_buy_signals['Date'], df_buy_signals['Adj Close'], label='Buy Signal', marker='^', color='green', alpha=1)
    plt.scatter(df_sell_signals['Date'], df_sell_signals['Adj Close'], label='Sell Signal', marker='v', color='red', alpha=1)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(signal)
    plt.legend()
    plt.grid(True)
    plt.show()

    profit1, profit_percentage1 = check_profit(df_filter['Adj Close'],df_filter[signal_encoded],1000)
    print("TI : profit",profit1,'profit_percentage',profit_percentage1)
    if ind_size==1:
        X_test_scaled = scaler.transform(df_filter[['Adj Close', indicators[0]]])
    if ind_size>=2:
        X_test_scaled = scaler.transform(df_filter[['Adj Close', indicators[0],indicators[1]]])

    #
    ylr_pred = modellr.predict(X_test_scaled)
    yrf_pred = modelrf.predict(X_test_scaled)
    ysvm_pred = modelsvm.predict(X_test_scaled)

    profit2, profit_percentage2 = check_profit(df_filter['Adj Close'],pd.Series(ylr_pred),1000)
    print("ML LR  : profit",profit2,'profit_percentage',profit_percentage2)

    profit2, profit_percentage2 = check_profit(df_filter['Adj Close'],pd.Series(yrf_pred),1000)
    print("ML RF  : profit",profit2,'profit_percentage',profit_percentage2)

    profit2, profit_percentage2 = check_profit(df_filter['Adj Close'],pd.Series(ysvm_pred),1000)
    print("ML SVM : profit",profit2,'profit_percentage',profit_percentage2)


    plt.figure(figsize=(14, 7))
    plt.plot(df_filter['Date'], df_filter['Adj Close'], label='Adj Close', color='blue')
    if ind_size>0:
        plt.plot(df_filter['Date'], df_filter[indicators[0]], label=indicators[0], color='orange')
    if ind_size>1:
        plt.plot(df_filter['Date'], df_filter[indicators[1]], label=indicators[1], color='orange')
    df_buy_signals = df_filter[ylr_pred == 1]
    df_sell_signals = df_filter[ylr_pred == -1]
    plt.scatter(df_buy_signals['Date'], df_buy_signals['Adj Close'], label='Buy Signal', marker='^', color='green', alpha=1)
    plt.scatter(df_sell_signals['Date'], df_sell_signals['Adj Close'], label='Sell Signal', marker='v', color='red', alpha=1)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title("LR "+signal)
    plt.legend()
    plt.grid(True)
    plt.show()


    plt.figure(figsize=(14, 7))
    plt.plot(df_filter['Date'], df_filter['Adj Close'], label='Adj Close', color='blue')
    if ind_size>0:
        plt.plot(df_filter['Date'], df_filter[indicators[0]], label=indicators[0], color='orange')
    if ind_size>1:
        plt.plot(df_filter['Date'], df_filter[indicators[1]], label=indicators[1], color='orange')
    df_buy_signals = df_filter[yrf_pred == 1]
    df_sell_signals = df_filter[yrf_pred == -1]
    plt.scatter(df_buy_signals['Date'], df_buy_signals['Adj Close'], label='Buy Signal', marker='^', color='green', alpha=1)
    plt.scatter(df_sell_signals['Date'], df_sell_signals['Adj Close'], label='Sell Signal', marker='v', color='red', alpha=1)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title("RF "+signal)
    plt.legend()
    plt.grid(True)
    plt.show()


    plt.figure(figsize=(14, 7))
    plt.plot(df_filter['Date'], df_filter['Adj Close'], label='Adj Close', color='blue')
    if ind_size>0:
        plt.plot(df_filter['Date'], df_filter[indicators[0]], label=indicators[0], color='orange')
    if ind_size>1:
        plt.plot(df_filter['Date'], df_filter[indicators[1]], label=indicators[1], color='orange')
    df_buy_signals = df_filter[ysvm_pred == 1]
    df_sell_signals = df_filter[ysvm_pred == -1]
    plt.scatter(df_buy_signals['Date'], df_buy_signals['Adj Close'], label='Buy Signal', marker='^', color='green', alpha=1)
    plt.scatter(df_sell_signals['Date'], df_sell_signals['Adj Close'], label='Sell Signal', marker='v', color='red', alpha=1)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title("SVM "+signal)
    plt.legend()
    plt.grid(True)
    plt.show()


sd = '1887-12-31'
ed = '2023-12-31'
data = yf.download("AAPL" ,start=sd, end=ed)
data.to_csv("AAPL.csv")
df = pd.read_csv("AAPL.csv")

pd_sd = pd.to_datetime(sd)
pd_ed = pd.to_datetime(ed)

stock_df = df[['Date', 'Adj Close','Volume']]
stock_df['Date'] = pd.to_datetime(stock_df['Date'])


stock_df = calculate_sma(stock_df,28)
stock_df = calculate_bollinger_bands(stock_df,window=28, num_std_dev=2)
stock_df = calculate_obv(stock_df)

stock_df = calculate_signals(stock_df)
stock_df = signal_map(stock_df)
stock_df.dropna(inplace=True)

df_train = stock_df[(stock_df['Date'] >= pd_sd) & (stock_df['Date'] <= pd_ed)]
df_filter = stock_df[(stock_df['Date'] >= pd.to_datetime('2023-01-01'))]

print("==============SMA================")
process_indicator(df_filter,["SMA"],"SignalSMA","SignalSMA_Encoded")
print("==============Bollinger Band Start ================")
process_indicator(df_filter,["Bollinger Lower Band","Bollinger Upper Band"],"SignalBB","SignalBB_Encoded")
print("==============OBV Start ================")
process_indicator(df_filter,["OBV"],"SignalOBV","SignalOBV_Encoded")