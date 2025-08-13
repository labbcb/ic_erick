
"""quickstart-docker-2: A Flower / PyTorch app."""

import os

import numpy as np
import keras
from keras import layers
from keras import regularizers
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

'''
def load_model():
    model = keras.Sequential(
        [
            keras.Input(shape=(2,)),
            layers.Dense(100, activation='relu'),
            layers.Dense(100, activation='relu'),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile("SGD", "binary_crossentropy", metrics=["accuracy"])
    return model
'''
def load_model():
    model = keras.Sequential(
        [
            keras.Input(shape=(23,)),
            layers.Dense(10, activation='relu'),
            layers.Dense(10, activation='relu'),
            #layers.Dense(20, activation='relu', kernel_regularizer = regularizers.L1(1e-2)),
            #layers.Dense(20, activation='relu', kernel_regularizer = regularizers.L1(1e-2)),
            layers.Dense(4, activation="softmax"),
        ]
    )
    model.compile(keras.optimizers.SGD(learning_rate = 5e-4), "sparse_categorical_crossentropy", metrics=["accuracy"])
    return(model)

def load_autoencoder_model(input_size: int, encoded_size: int): 
    model = keras.Sequential(
        [
            keras.Input(shape=(input_size,)),
            layers.Dense(encoded_size, activation='relu', activity_regularizer = regularizers.L1(1e-5)),
            layers.Dense(input_size, activation="sigmoid"),
        ]
    )
    model.compile(keras.optimizers.SGD(learning_rate = 1e-2), loss = 'mse')
    return(model)

def load_data(tipo_dados, pre_processamento, oversample):
    if tipo_dados == 'fedpca' and pre_processamento == 'No':
       if oversample == 'Yes':
          x_train = pd.read_csv("/app/x_train_pca.csv", header = None)
          x_test = pd.read_csv("/app/x_test_pca.csv", header = None)
          y_train = pd.read_csv("/mnt/fl_conj_treino_cliente.csv")['subtipo2']
          y_test = pd.read_csv("/mnt/fl_conj_teste_cliente.csv")['subtipo2']
       elif oversample == 'No':
          if os.path.exists("/app/x_train_pca_resampled.csv"): 
             x_train = pd.read_csv("/app/x_train_pca_resampled.csv", header = None)
             x_test = pd.read_csv("/app/x_test_pca.csv", header = None)
             y_train = pd.read_csv("/app/y_train_pca_resampled.csv", header = None)
             y_test = pd.read_csv("/mnt/fl_conj_teste_cliente.csv")['subtipo2']
          else:
             x_train = pd.read_csv("/app/x_train_pca.csv", header = None)
             x_test = pd.read_csv("/app/x_test_pca.csv", header = None)
             y_train = pd.read_csv("/mnt/fl_conj_treino_cliente.csv")['subtipo2']
             y_test = pd.read_csv("/mnt/fl_conj_teste_cliente.csv")['subtipo2']
    elif tipo_dados == 'autoencoder' and pre_processamento == 'No':
       if oversample == 'Yes':
          x_train = pd.read_csv("/app/x_train_ac.csv", header = None)
          x_test = pd.read_csv("/app/x_test_ac.csv", header = None)
          y_train = pd.read_csv("/mnt/fl_conj_treino_cliente.csv")['subtipo2']
          y_test = pd.read_csv("/mnt/fl_conj_teste_cliente.csv")['subtipo2']
       elif oversample == 'No':
          if os.path.exists("/app/x_train_ac_resampled.csv"):
             x_train = pd.read_csv("/app/x_train_ac_resampled.csv", header = None)
             x_test = pd.read_csv("/app/x_test_ac.csv", header = None)
             y_train = pd.read_csv("/app/y_train_ac_resampled.csv", header = None)
             y_test = pd.read_csv("/mnt/fl_conj_teste_cliente.csv")['subtipo2']
          else:
             x_train = pd.read_csv("/app/x_train_ac.csv", header = None)
             x_test = pd.read_csv("/app/x_test_ac.csv", header = None)
             y_train = pd.read_csv("/mnt/fl_conj_treino_cliente.csv")['subtipo2']
             y_test = pd.read_csv("/mnt/fl_conj_teste_cliente.csv")['subtipo2']
    else:
       dados_treino = pd.read_csv("/mnt/fl_conj_treino_cliente.csv")
       dados_teste = pd.read_csv("/mnt/fl_conj_teste_cliente.csv")
       x_train, y_train = dados_treino.iloc[:,5:], dados_treino['subtipo2']
       x_test, y_test = dados_teste.iloc[:,5:], dados_teste['subtipo2']
    ee = LabelEncoder()
    y_train = ee.fit_transform(y_train)
    y_test = ee.fit_transform(y_test)
    return x_train, y_train, x_test, y_test

def get_model(penalty: str, C: float, solver: str, max_iter: int):

    return LogisticRegression(
        penalty=penalty,
        C = C,
        random_state = 1,
        solver = solver,
        max_iter = max_iter,
        warm_start = True,
    )


def get_model_params(model):
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [model.coef_]
    return params


def set_model_params(model, params):
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model):
    n_classes = 4
    n_features = 23
   
    model.classes_ = np.array([i for i in range(n_classes)])
    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))
