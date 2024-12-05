import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, LSTM, Reshape, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error
from scipy.stats import linregress
import random
import statsmodels.api as sm
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import tempfile
from pathlib import Path
import tsmoothie
import os
from pymongo import MongoClient
from stock_plots import generate_plot


client = MongoClient('localhost', 27017)
db = client['Stocks']
coll_list = db.list_collection_names()
kalman_smoother = tsmoothie.KalmanSmoother(component='level_trend', component_noise={'level':0.1, 'trend':0.1})
timestep = 10
input_shape_LSTM = (timestep, 1)
num_filters_space = [16, 32, 64, 128]
num_layer = [1, 2, 3]
kernel_size_space = [2, 3, 4, 5]
dense_neuron_space=[49, 56, 63, 64]
dropout_rate_space = [0.2, 0.4, 0.6]
models = os.listdir('models')
correction_models = os.listdir('correction_models')
correction_models_replacements = {'_correction':''}
database_replacements = {'_N1Close_500.keras':'', '_N1High_500.keras':'', '_N1Low_500.keras':'', '_N1Open_500.keras':''}
used = []

def log_transform(df):
    transformed_df = np.log(df)
    transformed_df = transformed_df.fillna(-1)
    return transformed_df

def kalman_filter(df):
    dataframe = pd.DataFrame(df)
    smoothed_data = pd.DataFrame()
    for column in dataframe.columns:
        smoothed_data[column] = pd.Series(kalman_smoother.smooth(dataframe[column]).smooth_data.flatten(), index=dataframe.index)
    return smoothed_data

def initialize_population_LSTM(pop_size):
    population = []
    for _ in range(pop_size):
        individual = {
            'num_filters': random.choice(num_filters_space),
            'num_layer_LSTM': random.choice(num_layer),
            'num_layer_Dense': random.choice(num_layer),
            'dropout_rate': random.choice(dropout_rate_space),
            'dense_neuron': random.choice(dense_neuron_space),
        }
        population.append(individual)
    return population

def create_LSTM_model(input_shape_LSTM, num_neurons, num_filters, num_layer_LSTM, num_layer_Dense, dropout_rate):
    model = Sequential()
    for _ in range(num_layer_LSTM):
        model.add(LSTM(num_filters, return_sequences=(_ < num_layer_LSTM - 1), input_shape=input_shape_LSTM))
    for _ in range(num_layer_Dense):
        model.add(Dense(num_neurons, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))    
    model.compile(optimizer=Adam(), loss='mae')
    return model

def fitness_function_LSTM(individual, error_feature, error_target):
    model = create_LSTM_model(input_shape_LSTM, individual['dense_neuron'], individual['num_filters'], individual['num_layer_LSTM'], individual['num_layer_Dense'],individual['dropout_rate'])    
    model.fit(error_feature, error_target, epochs=10, batch_size=32)    
    val_mae = model.evaluate(error_feature, error_target) 
    return val_mae

def mutation_LSTM(individual):
    mutation_key = random.choice(list(individual.keys()))
    if mutation_key == 'num_filters':
        individual[mutation_key] = random.choice(num_filters_space)
    elif mutation_key == 'num_layer_LSTM':
        individual[mutation_key] = random.choice(num_layer)
    elif mutation_key == 'num_layer_Dense':
        individual[mutation_key] = random.choice(num_layer)
    elif mutation_key == 'dropout_rate':
        individual[mutation_key] = random.choice(dropout_rate_space)
    elif mutation_key == 'dense_neuron':
        individual[mutation_key] = random.choice(dense_neuron_space)
    return individual

def crossover(parent1, parent2):
    child = {}
    for key in parent1.keys():
        child[key] = random.choice([parent1[key], parent2[key]])
    return child

def selection(population, fitness_scores):
    selected_indices = np.argsort(fitness_scores)[:2]
    return [population[i] for i in selected_indices]

def genetic_algorithm_LSTM(error_feature, error_target, pop_size, generations):
    population = initialize_population_LSTM(pop_size)
    for generation in range(generations):
        fitness_scores = [fitness_function_LSTM(ind, error_feature, error_target) for ind in population]
        print(f"Generation {generation+1} | Best Fitness: {min(fitness_scores)}")
        selected_parents = selection(population, fitness_scores)
        new_population = selected_parents[:]
        while len(new_population) < pop_size:
            parent1, parent2 = random.sample(selected_parents, 2)
            child = crossover(parent1, parent2)
            if random.random() < 0.1:  # Mutation probability
                child = mutation_LSTM(child)
            new_population.append(child)
        population = new_population
    final_fitness_scores = [fitness_function_LSTM(ind, error_feature, error_target) for ind in population]
    best_individual = population[np.argmin(final_fitness_scores)]
    return best_individual

def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        target = data.iloc[i + sequence_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

def multi_replace(document, replacements):
    for pos in range(len(document)):
        for search, replace in replacements.items():
            document[pos] = document[pos].replace(search, replace)
    return document


correction_models = multi_replace(correction_models,correction_models_replacements)
for file in models:
    if file not in correction_models:
        print(f'================ Training correction model for {file} ================\n')
        print('Loading Dataset....')
        database = multi_replace([file],database_replacements)
        coll = db[database[0]]
        cursor = coll.find()
        data = list(cursor)
        print(f'================ Dataset for {database[0]} ==============')
        df = pd.DataFrame(data)
        print(df.head())
        df = df.drop('_id', axis=1)
        df = df.set_index("Date")
        features = ['Open','High', 'Low', 'Close', 'Adj Close']
        print('Preparing data for train and test.....')
        target = file[0].replace(f'{database[0]}_','').replace('_500.keras','')
        train = df.loc[:'2023-10-10']
        train_dates = train.index.strftime('%Y-%m-%d')
        dncp_train = [i for i in range(len(train_dates))]
        df_test = df.loc['2023-10-10':]
        test_dates = df_test.index.strftime('%Y-%m-%d')
        dncp_test = [i for i in range(len(test_dates))]
        X_train = train[features]
        Y_train = train[target]
        X_test = df_test[features]
        Y_test = df_test[target]
        X_train = log_transform(X_train)
        Y_train = log_transform(Y_train)
        X_test = log_transform(X_test)
        Y_test = log_transform(Y_test)
        X_train = kalman_filter(X_train)
        Y_train = kalman_filter(Y_train)
        X_test = kalman_filter(X_test)
        Y_test = kalman_filter(Y_test)
        scaler_features = RobustScaler().fit(X_train)
        scaler_target = RobustScaler().fit(Y_train)
        X_train = pd.DataFrame(scaler_features.transform(X_train), columns = features)
        Y_train = pd.DataFrame(scaler_target.transform(Y_train), columns = [target])
        X_test = pd.DataFrame(scaler_features.transform(X_test), columns = features)
        Y_test = pd.DataFrame(scaler_target.transform(Y_test), columns = [target])
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        Y_train = tf.convert_to_tensor(Y_train, dtype=tf.float32)
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
        Y_test = tf.convert_to_tensor(Y_test, dtype=tf.float32)
        print(f'Loading {file}....')
        best_model = tf.keras.models.load_model(f'models/{file[0]}')
        print(f'Predicting training data...')
        y_pred_train = best_model.predict(X_train)
        Y_train = scaler_target.inverse_transform(Y_train)
        Y_train = np.exp(Y_train)
        y_pred_train = scaler_target.inverse_transform(y_pred_train)
        y_pred_train = np.exp(y_pred_train)
        train_period_errors = Y_train/y_pred_train
        print(f'Preparing data for correction model')
        error_df = pd.DataFrame(data=train_period_errors, index= train.index.strftime('%Y-%m-%d'), columns= ['Error Ratio'])
        error_feature, error_target = create_sequences(error_df, sequence_length= timestep)
        print(f'Optimizing Hyperparameter for Correction model....')
        best_hyperparameters_LSTM = genetic_algorithm_LSTM(error_feature, error_target, pop_size=10, generations=5)
        print(f"\nBest Hyperparameters for Correction Model: {best_hyperparameters_LSTM}")
        print('Initializing Correction model with best hyperparameter...')
        correction_model = Sequential()
        for _ in range(int(best_hyperparameters_LSTM['num_layer_LSTM'])):
            correction_model.add(LSTM(int(best_hyperparameters_LSTM['num_filters']), return_sequences=(_ < int(best_hyperparameters_LSTM['num_layer_LSTM']) - 1), input_shape=input_shape_LSTM))
        for _ in range(int(best_hyperparameters_LSTM['num_layer_Dense'])):
            correction_model.add(Dense(int(best_hyperparameters_LSTM['dense_neuron']), activation='relu'))
        correction_model.add(Dropout(int(best_hyperparameters_LSTM['dropout_rate'])))
        correction_model.add(Dense(1))    
        correction_model.compile(optimizer=Adam(), loss='mae')
        print('Training Correction model...')
        correction_model.fit(error_feature, error_target, epochs = 40, batch_size=32)
        print('Saving Correction model...')
        correction_model.save(f'models/{file[0].replace('.keras','')}_correction.keras')
        print('Predicting Training error...')
        predicted_error = correction_model.predict(error_feature)
        corrected_train_prediction = y_pred_train[timestep:]*predicted_error
        Y_test = scaler_target.inverse_transform(Y_test)
        Y_test = np.exp(Y_test)
        Corrected_test_predictions = []
        test_raw_predictions = []
        i = 0
        print('Running model on day by day method...')
        for test_data in X_test:
            test_data = tf.reshape(test_data,(1,test_data.shape[0],1))
            pred = best_model.predict(test_data) # Predicting value for the Validation period
            pred = scaler_target.inverse_transform(pred)
            pred = np.exp(pred)
            past_error = error_df.tail(timestep).to_numpy() # Taking errors for the last 10 days excluding current error
            past_error = past_error.reshape((1, past_error.shape[0], 1))
            current_error = correction_model.predict(past_error) # Predicting the Current error
            LSTM_Corrected = pred*current_error
            Corrected_test_predictions.append(LSTM_Corrected[0][0])
            test_raw_predictions.append(pred)
            # Adding Current error value to the dataframe
            error_df.loc[test_dates[i]] = Y_test[i]/pred[0]
            i=i+1
        print('Plotting Graph for test period with correction...\n')
        plot_image=generate_plot(predicted_targets=Corrected_test_predictions,actual_targets=[value[0] for value in Y_test],dates=test_dates,target=target, stock_name = file[0].replace('.keras','')+'_corrected')
    else:
        print(f'=============== Correction model for {file} already exist ===============\n')