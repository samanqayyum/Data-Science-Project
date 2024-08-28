
"""
Created on Sat Jun 15 22:42:57 2024

@author: Saman Qayyum
"""

import warnings
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Function to calculate the Simple Moving Average (SMA)


def calculate_sma(ds, window):
    """
    Calculate Simple Moving Average (SMA) for the 'Adj Close' column in dataframe.

    This function computes the SMA using a rolling window of specified size
    and adds result as a new column 'SMA' in the input DataFrame.

    Parameters:
    ds (pd.DataFrame): DataFrame containing the stock data with an 'Adj Close' column.
    window (int): The number of periods over which to calculate the SMA.

    Returns:
    pd.DataFrame: DataFrame with an additional column 'SMA' containing
                  computed SMA values.
    """

    # Compute the Simple Moving Average (SMA) and add it as a new column
    ds["SMA"] = ds['Adj Close'].rolling(window=window).mean()

    return ds

# Function to calculate Bollinger Bands


def calculate_bollinger_bands(ds, window, num_std_dev):
    """
    Calculate Bollinger Bands for the 'Adj Close' column in the dataframe.

    Bollinger Bands consist of a middle band (SMA), an upper band, and a lower
    band. The upper and lower bands are determined based on rolling standard
    deviation and a specified number of standard deviations from middle band.

    Parameters:
    ds (pd.DataFrame): DataFrame containing the stock data with an 'Adj Close' column.
    window (int): The number of periods over which to calculate rolling statistics.
    num_std_dev (float): The number of standard deviations to determine the width of bands.

    Returns:
    pd.DataFrame: DataFrame with additional columns 'Bollinger Middle Band',
                 'Bollinger Upper Band', and 'Bollinger Lower Band' containing
                  computed Bollinger Bands.
    """

    # Compute the middle band as the Simple Moving Average (SMA)
    ds['Bollinger Middle Band'] = ds['Adj Close'].rolling(window=window).mean()

    # Compute rolling standard deviation
    rolling_std = ds['Adj Close'].rolling(window=window).std()

    # Compute upper Bollinger band
    ds['Bollinger Upper Band'] = ds['Bollinger Middle Band'] + \
        (rolling_std * num_std_dev)

    # Compute lower Bollinger Band
    ds['Bollinger Lower Band'] = ds['Bollinger Middle Band'] - \
        (rolling_std * num_std_dev)

    return ds

# Function to calculate On-Balance Volume (OBV)


def calculate_obv(ds):
    """
    Calculate the On-Balance Volume (OBV) for the given DataFrame.

    On-Balance Volume (OBV) is an indicator that uses volume flow to predict
    changes in stock price. The OBV is computed by adding or subtracting volume
    from previous OBV value based on whether the price has increased or decreased
    compared to the previous day.

    Parameters:
    ds (pd.DataFrame): DataFrame containing stock data with 'Adj Close' and 'Volume'
                       columns.

    Returns:
    pd.DataFrame: DataFrame with an additional column 'OBV' containing the computed
                  On-Balance Volume values.
    """

    # Initialize OBV list with 0
    obv = [0]

    # Calculate OBV for each day
    for i in range(1, len(ds)):
        if ds['Adj Close'][i] > ds['Adj Close'][i - 1]:
            # If price increased, add volume to OBV
            obv.append(obv[-1] + ds['Volume'][i])
        elif ds['Adj Close'][i] < ds['Adj Close'][i - 1]:
            # If price decreased, subtract volume from OBV
            obv.append(obv[-1] - ds['Volume'][i])
        else:
            # If price remained the same, OBV stays unchanged
            obv.append(obv[-1])

    # Add OBV to dataframe
    ds['OBV'] = obv

    return ds

# Function to calculate trading signals based on technical indicators


def calculate_signals(ds):
    """
    Generate trading signals based on technical indicators: Simple Moving Average
    (SMA), Bollinger Bands (BB), and On-Balance Volume (OBV).

    The function adds columns to the DataFrame for each signal type: 'SignalBB',
    'SignalOBV', and 'SignalSMA'. Signals are generated based on the following rules:
    - SMA Signal: 'Buy' if the Adj Close is below the SMA, 'Sell' if above,
      otherwise 'Neutral'.
    - Bollinger Bands Signal: 'Buy' if the Adj Close is below the lower band, 'Sell'
      if above the upper band, otherwise 'Neutral'.
    - OBV Signal: 'Buy' if OBV is increasing, 'Sell' if decreasing, otherwise 'Neutral'.

    Parameters:
    ds (pd.DataFrame): DataFrame containing the stock data with 'Adj Close', 'SMA',
                      'Bollinger Upper Band', 'Bollinger Lower Band', and 'OBV' columns.

    Returns:
    pd.DataFrame: DataFrame with additional columns 'SignalBB', 'SignalOBV', and
                 'SignalSMA' containing the computed signals.
    """

    # Initialize signal columns for SMA, BB, OBV
    ds['SignalSMA'] = 'Neutral'
    ds['SignalBB'] = 'Neutral'
    ds['SignalOBV'] = 'Neutral'

    # Loop through each day to generate signals
    for i in range(0, len(ds)):
        # SMA signal
        if ds['Adj Close'].iloc[i] < ds['SMA'].iloc[i]:
            ds.at[i, 'SignalSMA'] = 'Buy'
        elif ds['Adj Close'].iloc[i] > ds['SMA'].iloc[i]:
            ds.at[i, 'SignalSMA'] = 'Sell'
        else:
            ds.at[i, 'SignalSMA'] = 'Neutral'

        # Bollinger Bands signal
        if ds['Adj Close'].iloc[i] < ds['Bollinger Lower Band'].iloc[i]:
            ds.at[i, 'SignalBB'] = 'Buy'
        elif ds['Adj Close'].iloc[i] > ds['Bollinger Upper Band'].iloc[i]:
            ds.at[i, 'SignalBB'] = 'Sell'
        else:
            ds.at[i, 'SignalBB'] = 'Neutral'

        # OBV signal
        if i == 1:
            ds.at[i, 'SignalOBV'] = 'Neutral'
        if ds['OBV'].iloc[i] > ds['OBV'].iloc[i - 1]:
            ds.at[i, 'SignalOBV'] = 'Buy'
        elif ds['OBV'].iloc[i] < ds['OBV'].iloc[i - 1]:
            ds.at[i, 'SignalOBV'] = 'Sell'
        else:
            ds.at[i, 'SignalOBV'] = 'Neutral'

    return ds

# Function to encode trading signals as numerical values


def signal_map(df):
    """
    Encode trading signals as numerical values and save DataFrame to a CSV file.

    This function maps the trading signals 'Buy', 'Sell', and 'Neutral' to
    numerical values and creates new columns in the DataFrame for these encoded
    signals. The encoded values are:
    - 'Buy' : 1
    - 'Sell': -1
    - 'Neutral': 0

    The function also saves the DataFrame with the encoded signals to a CSV file
    named "test.csv".

    Parameters:
    df (pd.DataFrame): DataFrame containing the trading signals in columns
                      'SignalSMA', 'SignalOBV', and 'SignalBB'.

    Returns:
    pd.DataFrame: DataFrame with additional columns 'SignalSMA_Encoded',
                 'SignalOBV_Encoded', and 'SignalBB_Encoded' containing the
                  encoded trading signals.
    """

    # Map trading signals to numerical values
    map = {'Buy': 1, 'Sell': -1, 'Neutral': 0}

    # Encode 'SignalSMA' column
    df['SignalSMA_Encoded'] = df['SignalSMA'].map(map)

    # Encode 'SignalBB' column
    df['SignalBB_Encoded'] = df['SignalBB'].map(map)

    # Encode 'SignalOBV' column
    df['SignalOBV_Encoded'] = df['SignalOBV'].map(map)

    # Save the DataFrame with encoded signals to a CSV file
    df.to_csv("test.csv")
    return df

# Function to evaluate profit based on trading signals


def check_profit(price, signal, start_cash):
    """
    Evaluate the profit based on trading signals and initial cash.

    This function simulates trading by executing buy or sell orders based on
    trading signals. It calculates the total profit and percentage profit from
    the initial cash after simulating trading over the given period.

    Parameters:
    price (pd.Series): Series containing the daily prices of the asset.
    signal (pd.Series): Series containing the trading signals where:
                        - 1 indicates a 'Buy' signal
                        - -1 indicates a 'Sell' signal
                        - 0 indicates 'Neutral' (no action).
    start_cash (float): The initial amount of cash available for trading.

    Returns:
           profit: The total profit from the trading.
           percentage profit: The percentage profit relative to the initial cash.
    """

    # Initialize cash
    current_cash = start_cash

    # Initialize stock holdings
    current_stock = 0

    # Loop through each day to simulate trading based on signals
    for i in range(len(price)):
        # Check if the signal for today is 'Buy' and we have cash available
        if signal.iloc[i] == 1 and current_cash > 0:
            # Buy as many stocks as possible with the available cash
            # Calculate the amount of stocks we can buy
            current_stock = current_cash / price.iloc[i]
            current_cash = 0  # Use up all available cash

        # Check if the signal for today is 'Sell' and we have stocks to sell
        elif signal.iloc[i] == -1 and current_stock > 0:
            # Sell stocks if signal is 'Sell' and stocks are available
            # Calculate cash obtained from selling stocks
            current_cash = current_stock * price.iloc[i]
            current_stock = 0  # Use up all stocks

    # Calculate final portfolio value
    final_value = current_cash + current_stock * price.iloc[-1]

    # Calculate profit and percentage profit
    # Total profit made from trading
    profit = final_value - start_cash

    # Profit percentage relative to initial cash
    percentage_profit = (profit / start_cash) * 100

    # Print initial cash amount, Final value of portfolio, Total profit, and \
    # percentage profit
    print("start_cash", start_cash, 'final_value', final_value,
          "profit", profit, "percentage_profit", percentage_profit)

    # Return the profit and percentage profit as a tuple
    return profit, percentage_profit

# Function to train and evaluate a Logistic Regression model


def logistic_regression(df, feature, target):
    """
    Train and evaluate a Logistic Regression model.

    This function prepares the data by selecting feature columns and the target
    variable, splits the data into training and test sets, standardizes the
    features, trains a Logistic Regression model, and evaluates its performance.

    Parameters:
    df (pd.DataFrame): DataFrame containing dataset with feature and target columns.
    feature : List of column names to be used as features.
    target: Column name for the target variable.

    Returns:
           - model (LogisticRegression): Trained Logistic Regression model.
           - scaler (StandardScaler): Fitted StandardScaler used for feature scaling.
    """

    # Create list of feature columns including 'Adj Close'
    cols = ['Adj Close']
    for f in feature:
        cols.append(f)

    # Extract feature columns and target variable from DataFrame
    features = df[cols]
    target = df[target]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=42)

    # Standardize the features to have mean 0 and variance 1
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the model
    model = LogisticRegression(max_iter=1000, random_state=42)

    # Define the hyperparameters to tune
    param_grid = {'C': [0.01, 0.1, 1],  # Regularization strength
                  'penalty': ['l2'],  # Regularization types
                  'solver': ['saga', 'liblinear', 'lbfgs', 'newton-cg']}  # Solvers

    # Initialize GridSearchCV to perform hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=5, scoring='f1_macro',
        n_jobs=-1, verbose=2)

    # Fit model using GridSearchCV on training data to find best hyperparameters
    grid_search.fit(X_train_scaled, y_train)

    # Print the best used hyperparameters
    print("Best parameters found: ", grid_search.best_params_)

    # Train the final model with the best parameters
    best_model = grid_search.best_estimator_

    # Fit the best model on the training data
    best_model.fit(X_train_scaled, y_train)

    # Predict labels for the test data
    predictions = best_model.predict(X_test_scaled)

    # Evaluate the model's performance by calculating accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Print the accuracy as a percentage
    print(f"Logistic Regression Model Accuracy: {accuracy * 100:.2f}%")

    # Print class distribution in training data
    print("Training class distribution:")
    print(y_train.value_counts())

    # Evaluate the model by printing confusion matrix
    print("Confusion Matrix:")
    conf_matrix = confusion_matrix(y_test, predictions)
    print(conf_matrix)

    # Evaluate the model by printing classification report
    print("Classification Report:")
    print(classification_report(y_test, predictions))

    # Create a heatmap
    # Set the size of the figure for the plot
    plt.figure(figsize=(8, 6))
    # Create a heatmap to visualize the confusion matrix
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    # Set the title of the plot
    plt.title('Confusion Matrix - Logistic Regression',
              fontsize=17, fontweight='bold', y=1.01)
    # Label the x-axis as 'Predicted Labels'
    plt.xlabel('Predicted Labels',  fontsize=15)
    # Label the y-axis as 'True Labels'
    plt.ylabel('True Labels',  fontsize=15)
    # Display the plot
    plt.show()

    # Return the trained model and the scaler used for feature scaling
    return best_model, scaler

# Function to train and evaluate a Random Forest Classifier model


def random_forest(df, feature, target):
    """
    Train and evaluate a Random Forest Classifier model.

    This function prepares the data by selecting feature columns and the target
    variable, splits the data into training and test sets, standardizes the
    features, trains a Random Forest Classifier model, and evaluates its
    performance based on accuracy.

    Parameters:
    df (pd.DataFrame): DataFrame containing dataset with feature and target columns.
    feature: List of column names to be used as features.
    target: Column name for the target variable.

    Returns:
           - model (RandomForestClassifier): Trained Random Forest Classifier model.
           - scaler (StandardScaler): Fitted StandardScaler used for feature scaling.
    """

    # Create list of feature columns including 'Adj Close'
    cols = ['Adj Close']
    for f in feature:
        cols.append(f)

    # Extract feature columns and target variable from DataFrame
    features = df[cols]
    target = df[target]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=42)

    # Standardize the features to have mean 0 and variance 1
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

   # Define the model
    model = RandomForestClassifier(random_state=42)

    # Define the hyperparameters to tune
    param_grid = {'n_estimators': [50, 100, 200],  # no of trees in forest
                  # Maximum depth of each tree
                  'max_depth': [None, 10, 20, 30],
                  'max_features': ['auto', 'sqrt', 'log2']}  # no of features for best split

    # Initialize GridSearchCV to perform hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=5, scoring='f1_macro',
        n_jobs=-1, verbose=2)

    # Fit model using GridSearchCV on the training data to find best hyperparameters
    grid_search.fit(X_train_scaled, y_train)

    # Print the best parameters
    print("Best parameters found: ", grid_search.best_params_)

    # Train the final model with the best parameters
    best_model = grid_search.best_estimator_

    # Fit the best model on the training data
    best_model.fit(X_train_scaled, y_train)

    # Evaluate on the test data
    predictions = best_model.predict(X_test_scaled)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, predictions)

    # Print the accuracy as a percentage
    print(f"Random Forest Model Accuracy: {accuracy * 100:.2f}%")

    # Evaluate the model by printing confusion matrix
    print("Confusion Matrix:")
    conf_matrix = confusion_matrix(y_test, predictions)
    print(conf_matrix)

    # Evaluate the model by printing classification report
    print("Classification Report:")
    print(classification_report(y_test, predictions))

    # Create a heatmap
    # Set the size of the figure for the plot
    plt.figure(figsize=(8, 6))
    # Create a heatmap to visualize the confusion matrix
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    # Set the title of the plot
    plt.title('Confusion Matrix - Random Forest',
              fontsize=17, fontweight='bold', y=1.01)
    # Label the x-axis as 'Predicted Labels'
    plt.xlabel('Predicted Labels',  fontsize=15)
    # Label the y-axis as 'True Labels'
    plt.ylabel('True Labels',  fontsize=15)
    # Display the plot
    plt.show()

    # Return the trained model and the scaler used for feature scaling
    return best_model, scaler

# Function to train and evaluate a Support Vector Machine (SVM) model


def support_vector_machine(df, feature, target):
    """
    Train and evaluate a Support Vector Machine (SVM) model.

    This function prepares the data by selecting feature columns and the target
    variable, splits the data into training and test sets, standardizes the
    features, trains a Support Vector Machine model with a linear kernel, and
    evaluates its performance based on accuracy.

    Parameters:
    df (pd.DataFrame): DataFrame containing dataset with feature and target columns.
    feature: List of column names to be used as features.
    target: Column name for the target variable.

    Returns:
           - model (SVC): Trained Support Vector Machine model.
           - scaler (StandardScaler): Fitted StandardScaler used for feature scaling.
    """

    # Prepare the list of feature columns including 'Adj Close'
    cols = ['Adj Close']
    for f in feature:
        cols.append(f)

    # Extract feature columns and target variable from DataFrame
    features = df[cols]
    target = df[target]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=42)

    # Standardize the features to have a mean of 0 and variance of 1
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the Support Vector Machine model with a linear kernel
    # defining parameter range
    param_grid = {'C': [0.0001, 0.001, 0.1, 1],  # Regularization parameter
                  'gamma': [1, 0.1, 0.01],  # Kernel coefficient
                  'kernel': ['linear', 'rbf']}  # kernel function

    # Initialize GridSearchCV to perform hyperparameter tuning
    grid = GridSearchCV(SVC(), param_grid, refit=True,
                        scoring='f1_macro', verbose=3)

    # Set the model to be used for grid search
    model = grid

    # Fit the model with the best parameters on the scaled training data
    model.fit(X_train_scaled, y_train)

    # print best parameter after tuning
    print(grid.best_params_)

    # print how our model looks after hyper-parameter tuning
    print(grid.best_estimator_)

    # Predict the target values for the test set
    predictions = model.predict(X_test_scaled)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, predictions)

    # Print the accuracy as a percentage
    print(f"Support Vector Machine Model Accuracy: {accuracy * 100:.2f}%")

    # Evaluate the model by printing confusion matrix
    print("Confusion Matrix:")
    conf_matrix = confusion_matrix(y_test, predictions)
    print(conf_matrix)

    # Evaluate the model by printing classification report
    print("Classification Report:")
    print(classification_report(y_test, predictions))

    # Create a heatmap
    # Set the size of the figure for the plot
    plt.figure(figsize=(8, 6))
    # Create a heatmap to visualize the confusion matrix
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    # Set the title of the plot
    plt.title('Confusion Matrix - Support Vector Machine',
              fontsize=17, fontweight='bold', y=1.01)
    # Label the x-axis as 'Predicted Labels'
    plt.xlabel('Predicted Labels',  fontsize=15)
    # Label the y-axis as 'True Labels'
    plt.ylabel('True Labels',  fontsize=15)
    # Display the plot
    plt.show()

    # Return the trained model and the scaler used for feature scaling
    return model, scaler

# Function to process indicators, train models, and evaluate performance


def process_indicator(df_train, df_filter, indicators, signal, signal_encoded):
    """
    Process technical indicators, train models, evaluate performance,
    and visualize results.

    This function performs the following tasks:
    1. Trains Logistic Regression, Random Forest, and Support Vector Machine models.
    2. Plots adjusted close prices, technical indicators, and trading signals.
    3. Evaluates profit based on trading signals and model predictions.
    4. Visualizes buy/sell signals based on predictions from each model.

    Parameters:
    df_train (pd.DataFrame): DataFrame containing training dataset with date,
                             adjusted close price, indicator values, and
                             trading signals.
    df_filter (pd.DataFrame): DataFrame containing filtered dataset with date,
                              adjusted close price, indicator values, and
                              trading signals.
    indicators: List of column names representing technical
                              indicators to be used and plotted.
    signal: Column name indicating trading signals ('Buy', 'Sell', 'Neutral').
    signal_encoded: Column name for encoded trading signals (numerical
                          representation of trading signals).

    Returns:
    None: The function performs plotting and prints results but does not
          return any value.
    """

    # Number of indicators used for training and plotting
    ind_size = len(indicators)

    # Train models using the provided indicators and signal encoding
    # Logistic Regression model
    modellr, scaler = logistic_regression(df_train, indicators, signal_encoded)
    # Random Forest model
    modelrf, _ = random_forest(df_train, indicators, signal_encoded)
    # Support Vector Machine model
    modelsvm, _ = support_vector_machine(df_train, indicators, signal_encoded)

    # Evaluate profit for the indicator-based strategy
    profit1, profit_percentage1 = check_profit(
        df_filter['Adj Close'], df_filter[signal_encoded], 1000)
    # Print profit and percentage for indicator-based strategy
    print("TI : profit", profit1, 'profit_percentage', profit_percentage1)

    # Predict and evaluate profits using machine learning models
    # Standardize features for predictions
    # If only one indicator is used, include 'Adj Close' and the single \
    # indicator for scaling
    if ind_size == 1:
        X_test_scaled = scaler.transform(
            df_filter[['Adj Close', indicators[0]]])
    # If two or more indicators are used, include 'Adj Close' and the first two \
    # indicators for scaling
    if ind_size >= 2:
        X_test_scaled = scaler.transform(
            df_filter[['Adj Close', indicators[0], indicators[1]]])

    # Logistic Regression model predictions
    ylr_pred = modellr.predict(X_test_scaled)
    # Random Forest model predictions
    yrf_pred = modelrf.predict(X_test_scaled)
    # Support Vector Machine model predictions
    ysvm_pred = modelsvm.predict(X_test_scaled)

    # Evaluate profit for Logistic Regression model predictions
    profit2, profit_percentage2 = check_profit(
        df_filter['Adj Close'], pd.Series(ylr_pred), 1000)
    # Print profit and percentage for Logistic Regression
    print("ML LR  : profit", profit2, 'profit_percentage', profit_percentage2)

    # Evaluate profit for Random Forest model predictions
    profit2, profit_percentage2 = check_profit(
        df_filter['Adj Close'], pd.Series(yrf_pred), 1000)
    # Print profit and percentage for Random Forest
    print("ML RF  : profit", profit2, 'profit_percentage', profit_percentage2)

    # Evaluate profit for Support Vector Machine model predictions
    profit2, profit_percentage2 = check_profit(
        df_filter['Adj Close'], pd.Series(ysvm_pred), 1000)
    # Print profit and percentage for Support Vector Machine
    print("ML SVM : profit", profit2, 'profit_percentage', profit_percentage2)

    # Check if first indicator is On-Balance Volume (OBV)
    if indicators[0] == "OBV":
        # Create a new figure for the OBV plot
        plt.figure(figsize=(14, 7))
        # Plot the OBV indicator against the date
        plt.plot(df_filter['Date'], df_filter[indicators[0]],
                 label=indicators[0], color='orange')
        # Label the x-axis as 'Date'
        plt.xlabel('Date', fontsize=17)
        # Label the y-axis as 'Price'
        plt.ylabel('Price', fontsize=17)
        # Set the title of the plot to "OBV"
        plt.title("ON-Balance Volume", fontsize=20, fontweight='bold', y=1.01)
        # Display the legend for the plot
        plt.legend(fontsize=14)
        # Enable the grid for the plot
        plt.grid(True)
        # Show the plot
        plt.show()

    # Create a new figure for plotting the adjusted close price and indicators
    plt.figure(figsize=(14, 7))
    # Plot the adjusted close price against the date in blue
    plt.plot(df_filter['Date'], df_filter['Adj Close'],
             label='Adj Close', color='blue')
    # Check if first indicator is not On-Balance Volume (OBV)
    if indicators[0] != "OBV":
        # If there is at least one indicator, plot the first indicator in orange
        if ind_size > 0:
            plt.plot(df_filter['Date'], df_filter[indicators[0]],
                     label=indicators[0], color='orange')
        # If there are two or more indicators, plot the second indicator in orange
        if ind_size > 1:
            plt.plot(df_filter['Date'], df_filter[indicators[1]],
                     label=indicators[1], color='orange')

    # Filter the DataFrame to get buy signals
    df_buy_signals = df_filter[df_filter[signal] == 'Buy']
    # Filter the DataFrame to get sell signals
    df_sell_signals = df_filter[df_filter[signal] == 'Sell']
    # Plot the buy signals as green upward triangles
    plt.scatter(df_buy_signals['Date'], df_buy_signals['Adj Close'],
                label='Buy Signal', marker='^', color='green', alpha=1)
    # Plot the sell signals as red downward triangles
    plt.scatter(df_sell_signals['Date'], df_sell_signals['Adj Close'],
                label='Sell Signal', marker='v', color='red', alpha=1)
    # Set the x-axis label to 'Date'
    plt.xlabel('Date', fontsize=17)
    # Set the y-axis label to 'Price'
    plt.ylabel('Price', fontsize=17)
    # Set the title of the plot to the name of the signal
    plt.title(signal, fontsize=20, fontweight='bold', y=1.01)
    # Display the legend for the plot
    plt.legend(fontsize=14)
    # Enable the grid for the plot
    plt.grid(True)
    # Show the plot
    plt.show()

    # Create a new figure for plotting with a specified size
    plt.figure(figsize=(14, 7))
    # Plot the adjusted close price against the date in blue
    plt.plot(df_filter['Date'], df_filter['Adj Close'],
             label='Adj Close', color='blue')
    # Check if the first indicator is not On-Balance Volume (OBV)
    if indicators[0] != "OBV":
        # If there is at least one indicator, plot first indicator in orange
        if ind_size > 0:
            plt.plot(df_filter['Date'], df_filter[indicators[0]],
                     label=indicators[0], color='orange')
        # If there are two or more indicators, plot second indicator in orange
        if ind_size > 1:
            plt.plot(df_filter['Date'], df_filter[indicators[1]],
                     label=indicators[1], color='orange')
    # Filter the DataFrame to get predicted buy signals from the LR model
    df_buy_signals = df_filter[ylr_pred == 1]
    # Filter the DataFrame to get predicted sell signals from the LR model
    df_sell_signals = df_filter[ylr_pred == -1]
    # Plot the buy signals as green upward triangles
    plt.scatter(df_buy_signals['Date'], df_buy_signals['Adj Close'],
                label='Buy Signal', marker='^', color='green', alpha=1)
    # Plot the sell signals as red downward triangles
    plt.scatter(df_sell_signals['Date'], df_sell_signals['Adj Close'],
                label='Sell Signal', marker='v', color='red', alpha=1)
    # Set the x-axis label to 'Date'
    plt.xlabel('Date', fontsize=17)
    # Set the y-axis label to 'Price'
    plt.ylabel('Price', fontsize=17)
    # Set the title of the plot to include the signal and indicate it is for LR
    plt.title("Logistic Regression " + signal,
              fontsize=20, fontweight='bold', y=1.01)
    # Display the legend for the plot
    plt.legend(fontsize=14)
    # Enable the grid for the plot
    plt.grid(True)
    # Show the plot
    plt.show()

    # Create a new figure for plotting with a specified size
    plt.figure(figsize=(14, 7))
    # Plot the adjusted close price against the date in blue
    plt.plot(df_filter['Date'], df_filter['Adj Close'],
             label='Adj Close', color='blue')
    # Check if the first indicator is not On-Balance Volume (OBV)
    if indicators[0] != "OBV":
        # If there is at least one indicator, plot first indicator in orange
        if ind_size > 0:
            plt.plot(df_filter['Date'], df_filter[indicators[0]],
                     label=indicators[0], color='orange')
        # If there are two or more indicators, plot second indicator in orange
        if ind_size > 1:
            plt.plot(df_filter['Date'], df_filter[indicators[1]],
                     label=indicators[1], color='orange')
    # Filter the DataFrame to get predicted buy signals from Random Forest model
    df_buy_signals = df_filter[yrf_pred == 1]
    # Filter the DataFrame to get predicted sell signals from Random Forest model
    df_sell_signals = df_filter[yrf_pred == -1]
    # Plot the buy signals as green upward triangles
    plt.scatter(df_buy_signals['Date'], df_buy_signals['Adj Close'],
                label='Buy Signal', marker='^', color='green', alpha=1)
    # Plot the sell signals as red downward triangles
    plt.scatter(df_sell_signals['Date'], df_sell_signals['Adj Close'],
                label='Sell Signal', marker='v', color='red', alpha=1)
    # Set the x-axis label to 'Date'
    plt.xlabel('Date', fontsize=17)
    # Set the y-axis label to 'Price'
    plt.ylabel('Price', fontsize=17)
    # Set the title of plot to include signal and indicate it is for RF model
    plt.title("Random Forest " + signal,
              fontsize=20, fontweight='bold', y=1.01)
    # Display the legend for the plot
    plt.legend(fontsize=14)
    # Enable the grid for the plot
    plt.grid(True)
    # Show the plot
    plt.show()

    # Create a new figure for plotting with a specified size
    plt.figure(figsize=(14, 7))
    # Plot the adjusted close price against the date in blue
    plt.plot(df_filter['Date'], df_filter['Adj Close'],
             label='Adj Close', color='blue')
    # Check if the first indicator is not On-Balance Volume (OBV)
    if indicators[0] != "OBV":
        # If there is at least one indicator, plot the first indicator in orange
        if ind_size > 0:
            plt.plot(df_filter['Date'], df_filter[indicators[0]],
                     label=indicators[0], color='orange')
        # If there are two or more indicators, plot the second indicator in orange
        if ind_size > 1:
            plt.plot(df_filter['Date'], df_filter[indicators[1]],
                     label=indicators[1], color='orange')
    # Filter DataFrame to get predicted buy signals from SVM model
    df_buy_signals = df_filter[ysvm_pred == 1]
    # Filter the DataFrame to get predicted sell signals from the SVM model
    df_sell_signals = df_filter[ysvm_pred == -1]
    # Plot the buy signals as green upward triangles
    plt.scatter(df_buy_signals['Date'], df_buy_signals['Adj Close'],
                label='Buy Signal', marker='^', color='green', alpha=1)
    # Plot the sell signals as red downward triangles
    plt.scatter(df_sell_signals['Date'], df_sell_signals['Adj Close'],
                label='Sell Signal', marker='v', color='red', alpha=1)
    # Set the x-axis label to 'Date'
    plt.xlabel('Date', fontsize=17)
    # Set the y-axis label to 'Price'
    plt.ylabel('Price', fontsize=17)
    # Set title of the plot to include the signal and indicate it is for SVM model
    plt.title("Support Vector Machine " + signal,
              fontsize=20, fontweight='bold', y=1.01)
    # Display the legend for the plot
    plt.legend(fontsize=14)
    # Enable the grid for the plot
    plt.grid(True)
    # Show the plot
    plt.show()


# Define the start and end dates for historical data
SD = '2021-12-31'  # Start date for historical data
ED = '2023-12-31'  # End date for historical data


# Download historical stock data for Apple Inc. (AAPL) from Yahoo Finance
data = yf.download("AAPL", start=SD, end=ED)
# Save the downloaded data to a CSV file
data.to_csv("AAPL.csv")
# Read the data back from the CSV file into a DataFrame
df = pd.read_csv("AAPL.csv")

# print the dataframe created from historical data
print(df)
# Display the first few rows of the DataFrame to verify the data
print(df.head())


# Convert date strings to datetime objects
# Convert the start date string to a datetime object
pd_SD = pd.to_datetime(SD)
# Convert the end date string to a datetime object
pd_ED = pd.to_datetime(ED)


# Prepare the dataframe with relevant columns and convert 'Date' to datetime
# Select relevant columns: Date, Adjusted Close Price, and Volume
stock_df = df[['Date', 'Adj Close', 'Volume']]
# Convert the 'Date' column to datetime format for easier manipulation
stock_df['Date'] = pd.to_datetime(stock_df['Date'])


# Calculate technical indicators
# Calculate the 28-day Simple Moving Average (SMA) and add it to the dataframe
stock_df = calculate_sma(stock_df, 28)
# Calculate Bollinger Bands with a 28-day window and 2 standard deviations
stock_df = calculate_bollinger_bands(stock_df, window=28, num_std_dev=2)
# Calculate On-Balance Volume (OBV) and add it to the dataframe
stock_df = calculate_obv(stock_df)
# Calculate trading signals based on indicators and add them to the dataframe
stock_df = calculate_signals(stock_df)


# Encode trading signals as numerical values for model training
stock_df = signal_map(stock_df)

# Remove rows with missing values resulting from calculation of indicators or signals
stock_df.dropna(inplace=True)


# Filter data for training and testing
# Create a training dataset from the filtered data within the date range specified \
# by pd_SD and pd_ED
df_train = stock_df[(stock_df['Date'] >= pd_SD) & (stock_df['Date'] <= pd_ED)]
# Create a filtered dataset for recent data starting from January 1, 2023, for
# testing or further analysis
df_filter = stock_df[(stock_df['Date'] >= pd.to_datetime('2023-01-01'))]


# Process and evaluate indicators and models
# Process and evaluate Simple Moving Average (SMA) indicator
print("==============SMA================")
# Call process_indicator to evaluate the performance of the SMA trading signals
# Pass 'SMA' as the indicator, 'SignalSMA' as the trading signal column, and- \
# 'SignalSMA_Encoded' as the encoded signal column
process_indicator(df_train, df_filter, [
                  "SMA"], "SignalSMA", "SignalSMA_Encoded")

# Process and evaluate Bollinger Band (BB) indicator
print("==============Bollinger Band Start ================")
# Call process_indicator to evaluate performance of Bollinger Bands trading signals
# Pass 'Bollinger Lower Band' and 'Bollinger Upper Band' as indicators, 'SignalBB' \
# as the trading signal column, and 'SignalBB_Encoded' as the encoded signal column
process_indicator(df_train, df_filter, [
                  "Bollinger Lower Band", "Bollinger Upper Band"], "SignalBB",
                  "SignalBB_Encoded")

# Process and evaluate On-Balance Volume (OBV) indicator
print("==============OBV Start ================")
# Call process_indicator to evaluate the performance of OBV trading signals
# Pass 'OBV' as the indicator, 'SignalOBV' as the trading signal column, and /
# 'SignalOBV_Encoded' as the encoded signal column
process_indicator(df_train, df_filter, [
                  "OBV"], "SignalOBV", "SignalOBV_Encoded")
