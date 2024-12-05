import yfinance
import pandas as pd
import numpy as np
from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client['Stocks']

def Present_stock_data():
    for i in db.list_collection_names():
        print(f"===== {i} =====")
        coll = db[i]
        cursor = coll.find()
        data = list(cursor)
        df = pd.DataFrame(data)
        print(df.head())
        
Present_stock_data()