import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
import tsmoothie
import os
from pymongo import MongoClient
from yield_calculator import TYC_calculate_signals_orders_yields, calculate_intraday_yields
from datetime import timedelta

kalman_smoother = tsmoothie.KalmanSmoother(component='level_trend', component_noise={'level':0.1, 'trend':0.1})
timestep = 10
client = MongoClient('localhost', 27017)
db = client['Stocks']
coll_list = db.list_collection_names()

def multi_replace(document, replacements):
    for pos in range(len(document)):
        for search, replace in replacements.items():
            document[pos] = document[pos].replace(search, replace)
    return document

def log_transform(df):
    transformed_df = np.log(df)
    transformed_df = transformed_df.fillna(-1)
    return transformed_df

def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        target = data.iloc[i + sequence_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

def kalman_filter(df):
    dataframe = pd.DataFrame(df)
    smoothed_data = pd.DataFrame()
    for column in dataframe.columns:
        smoothed_data[column] = pd.Series(kalman_smoother.smooth(dataframe[column]).smooth_data.flatten(), index=dataframe.index)
    return smoothed_data

database = input("Please Enter Stock Sticker :")
file = os.listdir('models')
replacements = {'_N1Open_500.keras':'', '_N1High_500.keras':'', '_N1Low_500.keras':'', '_N1Close_500.keras':''}
files = set(multi_replace(file, replacements))
if database in files:
    print(f'================== Results for {database} ================')
    print(f'Loading Database for {database}...')
    coll = db[database]
    cursor = coll.find()
    data = list(cursor)
    df = pd.DataFrame(data)
    df = df.drop('_id', axis=1)
    print(f'\nView of dataset...')
    print(df.head())
    df['DNCP'] = (df["Date"] - df.head(1)['Date']).dt.days + 1
    dohlcav_mpnxp_data = df[['Date','DNCP','Open']]
    dohlcav_mpnxp_data = dohlcav_mpnxp_data.rename(columns = {
        'DCP': 'DCP_date_current_period',
        'DNCP': 'DNCP_day_number_current_period',
        'OPCP':'OPCP_open_price_current_period'
    })
    train_end_date = '2023-10-10'
    df = df.set_index("Date")
    print('Preparing data for prediction...')
    features = ['Open','High', 'Low', 'Close', 'Adj Close']
    target = ['N1Open','N1High','N1Low','N1Close']
    models = ['_N1Open_500.keras', '_N1High_500.keras', '_N1Low_500.keras', '_N1Close_500.keras']
    correction_models = ['_N1Open_500_correction.keras', '_N1High_500_correction.keras', '_N1Low_500_correction.keras', '_N1Close_500_correction.keras']
    train = df.loc[:'2023-10-10']
    df_train = df.iloc[:int(len(train)*0.9)]
    df_val = df.iloc[int(len(train)*0.9):]
    df_test = df.loc['2023-10-10':]
    test_dates = df_test.index.strftime('%Y-%m-%d')
    train_dates = df_train[timestep:].index.strftime('%Y-%m-%d')
    test_predictions = []
    test_predictions.append([test_dates])
    train_predictions = []
    train_predictions.append([train_dates])
    y_train_pred = []
    y_test_pred = []
    final_predicted_data = []
    for i in range(len(target)):
        print(f'\nRunning model for {database} {target[i]}...')
        val_data = df_val[features]
        val_targets = df_val[target[i]]
        X_train = df_train[features]
        Y_train = df_train[target[i]]
        X_test = df_test[features]
        Y_test = df_test[target[i]]
        val_data = log_transform(val_data)
        val_targets = log_transform(val_targets)
        X_train = log_transform(X_train)
        Y_train = log_transform(Y_train)
        X_test = log_transform(X_test)
        Y_test = log_transform(Y_test)
        val_data = kalman_filter(val_data)
        val_targets = kalman_filter(val_targets)
        X_train = kalman_filter(X_train)
        Y_train = kalman_filter(Y_train)
        X_test = kalman_filter(X_test)
        Y_test = kalman_filter(Y_test)
        scaler_features = RobustScaler().fit(X_train)
        scaler_target = RobustScaler().fit(Y_train)
        val_data = pd.DataFrame(scaler_features.transform(val_data), columns = features)
        val_targets = pd.DataFrame(scaler_target.transform(val_targets), columns = [target[i]])
        X_train = pd.DataFrame(scaler_features.transform(X_train), columns = features)
        Y_train = pd.DataFrame(scaler_target.transform(Y_train), columns = [target[i]])
        X_test = pd.DataFrame(scaler_features.transform(X_test), columns = features)
        Y_test = pd.DataFrame(scaler_target.transform(Y_test), columns = [target[i]])
        val_data = tf.convert_to_tensor(val_data, dtype=tf.float32)
        val_targets = tf.convert_to_tensor(val_targets, dtype=tf.float32)
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        Y_train = tf.convert_to_tensor(Y_train, dtype=tf.float32)
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
        Y_test = tf.convert_to_tensor(Y_test, dtype=tf.float32)
        model = tf.keras.models.load_model(f'models\{database+models[i]}')
        print(f'Predicting training period data...')
        y_pred_train = model.predict(X_train)
        if target[i] == 'N1Close':
            y_train_pred.append(y_pred_train)
        Y_train = scaler_target.inverse_transform(Y_train)
        Y_train = np.exp(Y_train)
        y_pred_train = scaler_target.inverse_transform(y_pred_train)
        y_pred_train = np.exp(y_pred_train)
        train_period_errors = Y_train/y_pred_train
        print(f'Preparing data for correction model...')
        error_df = pd.DataFrame(data=train_period_errors, index= df_train.index.strftime('%Y-%m-%d'), columns= ['Error Ratio'])
        error_feature, error_target = create_sequences(error_df, sequence_length= timestep)
        correction_model = tf.keras.models.load_model(f'correction_models\{database+correction_models[i]}')
        print("Predicting error's for training period...")
        predicted_error = correction_model.predict(error_feature)
        corrected_train_prediction = y_pred_train[timestep:]*predicted_error
        train_predictions.append([corrected_train_prediction])
        Y_test = scaler_target.inverse_transform(Y_test)
        Y_test = np.exp(Y_test)
        Corrected_test_predictions = []
        test_raw_predictions = []
        j = 0
        print('Running model on day by day method...')
        for test_data in X_test:
            test_data = tf.reshape(test_data,(1,test_data.shape[0],1))
            pred = model.predict(test_data, verbose=0) # Predicting value for the Validation period
            pred = scaler_target.inverse_transform(pred)
            pred = np.exp(pred)
            past_error = error_df.tail(timestep).to_numpy() # Taking errors for the last 10 days excluding current error
            past_error = past_error.reshape((1, past_error.shape[0], 1))
            current_error = correction_model.predict(past_error, verbose=0) # Predicting the Current error
            LSTM_Corrected = pred*current_error
            Corrected_test_predictions.append(LSTM_Corrected[0][0])
            test_raw_predictions.append(pred)
            if target[i]=='N1Close':
                y_test_pred.append(LSTM_Corrected[0][0])
            # Adding Current error value to the dataframe
            error_df.loc[test_dates[j]] = Y_test[j]/pred[0]
            j=j+1
        final_predicted_data.append(Corrected_test_predictions[-1])
        test_predictions.append(Corrected_test_predictions)
    print(f'\nPredictions for Date: {(pd.to_datetime(df_test.tail(1).index)+timedelta(1)).to_list()[0]}:\n Open: {final_predicted_data[0]}\n High: {final_predicted_data[1]}\n Low: {final_predicted_data[2]}\n Close: {final_predicted_data[3]}')
else:
    print(f"========== Currently we don't have models for {database} ========== ")