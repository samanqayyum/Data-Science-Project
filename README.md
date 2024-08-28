# Project Metadata

## Project Title:
"Comparative Analysis of Machine Learning and Traditional Statistical Methods for Stock Price Forecasting" 

## Project Overview:
This project aims to predict stock price movements for Apple Inc. (AAPL) from Yahoo Finance using machine learning models and technical analysis. By analyzing historical stock data of two years, trading signals are generated to assist in decision-making for buying, selling, or holding stocks. The project employs Logistic Regression, Random Forest, and Support Vector Machines (SVM) models, and incorporates key technical indicators such as Simple Moving Average (SMA), Bollinger Bands, and On-Balance Volume (OBV). Hyperparameter tuning is used to optimize model performance. Models are evaluated based on predictive accuracy, profitability, and return on investment (ROI), highlighting their strengths. Visual graphs of buy and sell signals offer insights into strategy effectiveness. The analysis shows that while technical indicators offer valuable insights, machine learning models outperform in accuracy and profitability. This study demonstrates the benefits of combining advanced computational techniques with traditional indicators to enhance trading strategies, offering valuable guidance for investors and analysts.

## Projct Objectives:
The objective of this project is to perform a comparative analysis between machine learning models and traditional statistical methods to forecast stock price movements. This study seeks to identify which approach offers better predictive accuracy, profitability, and return on investment (ROI), providing valuable insights for improving trading strategies and decision-making in stock market investments.

## Project Structure:
1.	Data Collection: Our dataset is sourced from Yahoo Finance, containing daily historical stock prices for Apple Inc. (AAPL). This dataset spans over 2 years and includes essential attributes like open, close, high, and low prices, along with adjusted close prices and trading volume with 502 rows and 7 columns. The dataset is stored in CSV format for easy accessibility and analysis.
2.	Feature Engineering: The script calculates technical indicators (SMA, Bollinger Bands, OBV) and generates trading signals based on these indicators.
3.	Model Training and Evaluation: Logistic Regression, Random Forest, and Support Vector Machine models are trained on historical data. The performance of these models is evaluated using accuracy, confusion matrix, and classification report.
4.	Profit Calculation: The script assesses the profitability of trading strategies based on the generated trading signals and model predictions.

## Installations:
Following packages are installed:
1.	yfinance: Download historical market data
2.	pandas: Data manipulation and analysis
3.	matplotlib: Plotting graphs
4.	seaborn: Enhanced data visualization
5.	scikit-learn: Machine learning library for model training and evaluation

## Usage:
1. Clone the GitHub repository.
2. Ensure the CSV files (companyname.csv) are placed in the project directory.
3. Install required dependencies using pip install -r requirements.txt.
4. Run the DSProject.py script to perform data analysis and visualization tasks.

## Functions
1.	calculate_sma(df, window)
Calculates the Simple Moving Average (SMA) for (28 days) window and adds it to the DataFrame.
Parameters: df (DataFrame), window (integer)
Returns: Modified DataFrame with SMA column

2.	calculate_bollinger_bands(df, window, num_std_dev)
Calculates Bollinger Bands for a 28 window and number of standard deviations (set = 2).
Parameters: df (DataFrame), window (integer), num_std_dev (integer)
Returns: Modified DataFrame with Bollinger Bands columns

3.	calculate_obv(df)
Calculates On-Balance Volume (OBV) and adds it to the DataFrame.
Parameters: df (DataFrame)
Returns: Modified DataFrame with OBV column

4.	calculate_signals(df)
Generates trading signals based on SMA, Bollinger Bands, and OBV indicators.
Parameters: df (DataFrame)
Returns: Modified DataFrame with trading signal columns

5.	signal_map(df)
Encodes trading signals as numerical values for model training. Buy(1), Sell(-1) and Neutral(0)
Parameters: df (DataFrame)
Returns: Modified DataFrame with encoded signals

6.	logistic_regression(df, feature, target)
Trains and evaluates a Logistic Regression model.
Parameters: df (DataFrame), feature (list of feature column names), target (target column name)
Returns: Trained Logistic Regression model and scaler

7.	random_forest(df, feature, target)
Trains and evaluates a Random Forest model.
Parameters: df (DataFrame), feature (list of feature column names), target (target column name)
Returns: Trained Random Forest model and scaler

8.	support_vector_machine(df, feature, target)
Trains and evaluates a Support Vector Machine (SVM) model.
Parameters: df (DataFrame), feature (list of feature column names), target (target column name)
Returns: Trained SVM model and scaler

9.	process_indicator(df_train, df_filter, indicators, signal, signal_encoded)
Processes technical indicators, trains models, evaluates performance, and visualizes results.
Parameters:
df_train: Training dataset (DataFrame)
df_filter: Filtered dataset (DataFrame)
indicators: List of technical indicators
signal: Column name indicating trading signals ('Buy', 'Sell', 'Neutral')
signal_encoded: Column name for encoded trading signals

## Project Documentation: 
For a deeper understanding of the project methodology and findings, refer to the project report, FPR.docx. This document provides comprehensive insights into the project background, data management strategies, and detailed analyses of the results obtained. however, Ethical considerations, data management, and project background are documented in readme.md.

## Document Control:
1. Data Files: companyname.csv
2. Code Files: DSProject.py
3. Documentation: readme.md
4. Project Reports: FPR.docx
5. Version Control: GitHub repository available at https://github.com/samanqayyum/Data-Science-Project

## GitHub Repository
[Data Science Project GitHub Repository](https://github.com/samanqayyum/Data-Science-Project)


