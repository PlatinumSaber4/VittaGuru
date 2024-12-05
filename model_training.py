import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, LSTM, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping
import random
import tsmoothie
import os
from pymongo import MongoClient
from stock_plots import generate_plot


client = MongoClient('localhost', 27017)
db = client['Stocks']
coll_list = db.list_collection_names()
kalman_smoother = tsmoothie.KalmanSmoother(component='level_trend', component_noise={'level':0.1, 'trend':0.1})
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
num_filters_space = [16, 32, 64, 128]
num_layer_space = [2, 3, 4, 5]
kernel_size_space = [2, 3, 4, 5]
dense_neuron_space=[49, 56, 63, 64]
dropout_rate_space = [0.2, 0.4, 0.6]

def multi_replace(document, replacements):
    for pos in range(len(document)):
        for search, replace in replacements.items():
            document[pos] = document[pos].replace(search, replace)
    return document

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

def initialize_population(pop_size):
    population = []
    for _ in range(pop_size):
        individual = {
            'num_filters': random.choice(num_filters_space),
            'num_layer': random.choice(num_layer_space),
            'kernel_size': random.choice(kernel_size_space),
            'num_layer_LSTM': random.choice(num_layer_space),
            'LSTM_units': random.choice(dense_neuron_space),
            'dropout_rate': random.choice(dropout_rate_space),
            'num_layer_Dense': random.choice(num_layer_space),
            'dense_neuron': random.choice(dense_neuron_space)
        }
        population.append(individual)
    return population

def create_cnn_model(input_shape, num_neurons, num_layer, num_filters, num_layer_lstm, LSTM_units, kernel_size, dropout_rate, num_layer_dense):
    model = Sequential()
    for _ in range(num_layer):
        model.add(Conv1D(filters=num_filters, kernel_size=min(kernel_size, input_shape[0]), activation='relu',  input_shape=input_shape, padding='same'))
    model.add(Reshape((-1, num_filters)))
    for _ in range(num_layer_lstm):
        model.add(LSTM(LSTM_units, return_sequences=(_ < num_layer_lstm - 1)))        
    model.add(Flatten())
    for _ in range(num_layer_dense):
        model.add(Dense(num_neurons, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=Adam(), loss=Huber(), metrics=['mae'])
    return model

def fitness_function(individual, input_shape, X_train, Y_train, val_data, val_targets):
    model = create_cnn_model(input_shape, individual['dense_neuron'], individual['num_layer'], individual['num_filters'], individual['num_layer_LSTM'], individual['LSTM_units'], individual['kernel_size'],individual['dropout_rate'], individual['num_layer_Dense'])
    model.fit(X_train, Y_train, validation_data=(val_data, val_targets), callbacks= early_stopping, epochs=10, batch_size=32, verbose=0)
    val_loss, val_mae = model.evaluate(val_data,val_targets,verbose=0)
    return val_mae

def selection(population, fitness_scores):
    selected_indices = np.argsort(fitness_scores)[:2]
    return [population[i] for i in selected_indices]

def crossover(parent1, parent2):
    child = {}
    for key in parent1.keys():
        child[key] = random.choice([parent1[key], parent2[key]])
    return child

def mutation(individual):
    mutation_key = random.choice(list(individual.keys()))
    if mutation_key == 'num_filters':
        individual[mutation_key] = random.choice(num_filters_space)
    elif mutation_key == 'kernel_size':
        individual[mutation_key] = random.choice(kernel_size_space)
    elif mutation_key == 'num_layer':
        individual[mutation_key] = random.choice(kernel_size_space)
    elif mutation_key == 'num_layer_LSTM':
        individual[mutation_key] = random.choice(kernel_size_space)
    elif mutation_key == 'LSTM_units':
        individual[mutation_key] = random.choice(kernel_size_space)
    elif mutation_key == 'num_layer_Dense':
        individual[mutation_key] = random.choice(kernel_size_space)
    elif mutation_key == 'dropout_rate':
        individual[mutation_key] = random.choice(dropout_rate_space)
    elif mutation_key == 'dense_neuron':
        individual[mutation_key] = random.choice(dense_neuron_space)
    return individual

def genetic_algorithm(pop_size, generations, input_shape, X_train, Y_train, val_data, val_targets):
    population = initialize_population(pop_size)
    for generation in range(generations):
        fitness_scores = [fitness_function(ind, input_shape, X_train, Y_train, val_data, val_targets) for ind in population]
        print(f"Generation {generation+1} | Best Fitness: {min(fitness_scores)}")
        selected_parents = selection(population, fitness_scores)
        new_population = selected_parents[:]
        while len(new_population) < pop_size:
            parent1, parent2 = random.sample(selected_parents, 2)
            child = crossover(parent1, parent2)
            if random.random() < 0.1:  # Mutation probability
                child = mutation(child)
            new_population.append(child)
        population = new_population
    final_fitness_scores = [fitness_function(ind, input_shape, X_train, Y_train, val_data, val_targets) for ind in population]
    best_individual = population[np.argmin(final_fitness_scores)]
    return best_individual

def Train_model(best_hyperparameters, input_shape, X_train, Y_train, val_data, val_targets, X_test, Y_test):
    best_model = Sequential()
    for _ in range(best_hyperparameters['num_layer']):
        best_model.add(Conv1D(filters=best_hyperparameters['num_filters'], kernel_size=min(best_hyperparameters['num_filters'], input_shape[0]), activation='relu',  input_shape=input_shape, padding='same'))
    for _ in range(best_hyperparameters['num_layer_LSTM']):
        best_model.add(LSTM(best_hyperparameters['LSTM_units'], return_sequences=(_ < best_hyperparameters['num_layer_LSTM'] - 1)))        
    best_model.add(Flatten())
    for _ in range(best_hyperparameters['num_layer_Dense']):
        best_model.add(Dense(best_hyperparameters['dense_neuron'], activation='relu'))
    best_model.add(Dropout(best_hyperparameters['dropout_rate']))
    best_model.add(Dense(1))
    best_model.compile(optimizer=Adam(), loss=Huber(), metrics=['mae'])
    best_model.fit(X_train, Y_train, validation_data=(val_data, val_targets), callbacks= early_stopping, epochs=30, batch_size=32, verbose=0)
    y_pred = best_model.predict(X_test)
    test_loss = best_model.evaluate(X_test,Y_test)
    print(f"Loss on test Set: {test_loss}")
    return y_pred, best_model

file = os.listdir('models')
replacements = {'_N1Open_500.keras':'', '_N1High_500.keras':'', '_N1Low_500.keras':'', '_N1Close_500.keras':''}
files = set(multi_replace(file, replacements))
counter = 0
best_hyperparameters = None
while counter<5:
    for j in coll_list:
        if j not in files:
            genetic_counter = 0
            coll = db[j]
            cursor = coll.find()
            data = list(cursor)
            df = pd.DataFrame(data)
            df = df.drop('_id', axis=1)
            df = df.set_index("Date")
            features = ['Open','High', 'Low', 'Close', 'Adj Close']
            target = ['N1Open','N1High','N1Low','N1Close']
            train = df.loc[:'2023-10-10']
            df_train = df.iloc[:int(len(train)*0.9)]
            df_val = df.iloc[int(len(train)*0.9):]
            df_test = df.loc['2023-10-10':]
            for i in range(len(target)):
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
                input_shape = (X_train.shape[1], 1)
                print(f"\n================== {j} ====================\n")
                if genetic_counter<1:
                    best_hyperparameters = genetic_algorithm(pop_size=10, generations=10, input_shape = input_shape, X_train = X_train, Y_train = Y_train, val_data = val_data, val_targets = val_targets)
                    print(f"Best Hyperparameters: {best_hyperparameters}\n")
                    parameter_database = client['Hyperparameters']
                    parameter_coll = parameter_database[j]
                    best_hyperparameters['target'] = target[i]
                    parameter_coll.insert_one(best_hyperparameters)
                y_pred, model = Train_model(best_hyperparameters, input_shape, X_train, Y_train, val_data, val_targets, X_test, Y_test)
                Y_test_dates= df_test.index
                Y_test_dates = Y_test_dates.strftime('%Y-%m-%d')
                Y_test = scaler_target.inverse_transform(Y_test)
                Y_test = np.exp(Y_test)
                y_pred = scaler_target.inverse_transform(y_pred)
                y_pred = np.exp(y_pred)
                plot_image=generate_plot(predicted_targets=[value[0] for value in y_pred],actual_targets=[value[0] for value in Y_test],dates=Y_test_dates,target=target[i], stock_name = j)
                model.save(f'models/{j}_{target[i]}_500.keras')
                genetic_counter =+1
    counter =+1