import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import lag_plot
from pymongo import MongoClient
import os

files = os.listdir('plots')
client = MongoClient('localhost', 27017)
db = client['Stocks']
replacements = {'_Close_VS_Time.png':'', '_Coorelation_Plot.png':'', '_Decomposition_Plot.png':'', '_Lag_Plot.png':'', '_Stock_Plot.png':''}
Stocks = db.list_collection_names()

def multi_replace(document, replacements):
    for pos in range(len(document)):
        for search, replace in replacements.items():
            document[pos] = document[pos].replace(search, replace)
    return document

def Describe(df):
    print(df.describe())
    
def Time_close_price_plot(df,name):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Close'], label='Close Price')
    plt.title('Closing Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.savefig(f"plots/{name}_Close_VS_Time.png")

def Stock_plot(df, name):
    df = df.drop(['Date','_id'], axis=1)
    df.plot(figsize=(12, 8))
    plt.title('Stock Data Over Time')
    plt.xlabel('Date')
    plt.savefig(f"plots/{name}_Stock_Plot.png")

def Coorelation_plot(df, name):
    df = df.drop(['Date','_id'], axis=1)
    corr_matrix = df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig(f"plots/{name}_Coorelation_Plot.png")

def Adf_result(df):
    adf_result = adfuller(df['Close'])
    print(f'ADF Statistic: {adf_result[0]}')
    print(f'p-value: {adf_result[1]}')
    if adf_result[1] <= 0.05:
        print("\nThe data is stationary\n") #just a sample not to be used
    else:
        print("\nThe data is not stationary\n")

def Decomposition_plot(df, name):
    decomposition = sm.tsa.seasonal_decompose(df['Close'], model='multiplicative', period=365)
    decomposition.plot()
    plt.savefig(f"plots/{name}_Decomposition_Plot.png")

def Lag_plot(df, name):
    plt.figure(figsize=(6, 6))
    lag_plot(df['Close'])
    plt.title('Lag Plot')
    plt.savefig(f"plots/{name}_Lag_Plot.png")

def Stock_EDA(symbol):
    print(f"\n========================= {symbol} EDA =========================\n")
    coll = db[symbol]
    cursor = coll.find()
    data = list(cursor)
    df = pd.DataFrame(data)
    print(f"\n========== {symbol} Description ==========\n")
    Describe(df)
    print(f"\n========= {symbol} Close vs Time plot =========\n")
    Time_close_price_plot(df, symbol)
    print(f"\n========== {symbol} Overall plot ==========\n")
    Stock_plot(df, symbol)
    print(f"\n========== {symbol} Coorelation plot ==========\n")
    Coorelation_plot(df, symbol)
    print(f"\n========== {symbol} Stationary result ==========\n")
    Adf_result(df)
    print(f"\n========== {symbol} Decomposition plot ==========\n")
    Decomposition_plot(df, symbol)
    print(f"\n========== {symbol} Lag plot ==========\n")
    Lag_plot(df, symbol)

stock_list = set(multi_replace(files, replacements))
for Stock in Stocks:
    if Stock in stock_list:
        print(f"\n======== Aready have plots for {Stock} ========\n")
    else:
        Stock_EDA(Stock)