import yfinance
import pandas as pd
import numpy as np
from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client['Stocks']

def Feature_creation(df):
    df["N1Open"] = df['Open'].shift(-1)
    df["N1High"] = df['High'].shift(-1)
    df["N1Low"] = df['Low'].shift(-1)
    df["N1Close"] = df['Close'].shift(-1)
    df = df.iloc[:len(df)-1]
    return df

def Data_extarctor(symbol):
    if symbol in db.list_collection_names():
        print('====== Stock Data Already Exist ======')
    else:
        coll = db[symbol]
        stock_data = pd.DataFrame(yfinance.download(symbol))
        stock_data = Feature_creation(stock_data)
        stock_data.reset_index(inplace=True)
        print(f"===== {symbol} =====")
        print(stock_data.head())
        stock_data_dict = stock_data.to_dict(orient='records')
        #print(stock_data.to_dict())
        coll.insert_many(stock_data_dict)
        print('===== Stock Added To The Database Successfully ======')
        
stocks = [
    # Automobile Sector
    'MARUTI.NS',      # Maruti Suzuki India
    'TATAMOTORS.NS',  # Tata Motors
    'M&M.NS',         # Mahindra & Mahindra
    'HEROMOTOCO.NS',  # Hero MotoCorp
    'TVSMOTOR.NS',    # TVS Motor Company

    # Banking & Financial Sector
    'HDFCBANK.NS',    # HDFC Bank
    'ICICIBANK.NS',   # ICICI Bank
    'IDFCFIRSTB.NS',  # IDFC First Bank
    'AXISBANK.NS',    # Axis Bank
    'PFC.NS',         # Power Finance Corporation (PFC)

    # Pharmaceutical Sector
    'SUNPHARMA.NS',   # Sun Pharmaceuticals
    'DIVISLAB.NS',    # Divi's Laboratories
    'DRREDDY.NS',     # Dr. Reddy’s Laboratories
    'CIPLA.NS',       # Cipla
    'LUPIN.NS',       # Lupin

    # Infrastructure Sector
    'ADANIENT.NS',    # Adani Enterprises
    'GMRINFRA.NS',    # GMR Airports Infrastructure
    'KEC.NS',         # KEC International
    'LT.NS',          # Larsen & Toubro
    'NBCC.NS',        # NBCC (India)

    # Real Estate Sector
    'DLF.NS',         # DLF
    'GODREJPROP.NS',  # Godrej Properties
    'LODHA.NS',       # Macrotech Developers
    'OBEROIRLTY.NS',  # Oberoi Realty
    'PRESTIGE.NS'     # Prestige Estates
]
for Stock in stocks:
    Data_extarctor(Stock)