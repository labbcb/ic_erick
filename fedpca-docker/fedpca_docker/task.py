
"""quickstart-docker-2: A Flower / PyTorch app."""

import os


import keras
from keras import layers
from keras import regularizers
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
            keras.Input(shape=(60660,)),
            layers.Dense(20, activation='relu', kernel_regularizer = regularizers.L1(1e-1)),
            layers.Dense(20, activation='relu', kernel_regularizer = regularizers.L1(1e-1)),
            layers.Dense(4, activation="softmax"),
        ]
    )
    model.compile("SGD", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return(model)

def load_data():
    dados_treino = pd.read_csv("/mnt/fl_conj_treino_cliente.csv")
    dados_teste = pd.read_csv("/mnt/fl_conj_teste_cliente.csv")
    x_train, y_train = dados_treino.iloc[:,5:], dados_treino['subtipo2']
    x_test, y_test = dados_teste.iloc[:,5:], dados_teste['subtipo2']
    ee = LabelEncoder()
    y_train = ee.fit_transform(y_train)
    y_test = ee.transform(y_test)

    return x_train, y_train, x_test, y_test

'''
def load_data():
    dados_treino = pd.read_csv("/mnt/fl_conj_treino_cliente.csv")
    dados_teste = pd.read_csv("/mnt/fl_conj_valid_cliente.csv")
    x_train, y_train = dados_treino[['x1', 'x2']], dados_treino['Resposta']
    x_test, y_test = dados_teste[['x1', 'x2']], dados_teste['Resposta']

    return x_train, y_train, x_test, y_test

'''
