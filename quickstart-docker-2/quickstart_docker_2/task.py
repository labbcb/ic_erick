
"""quickstart-docker-2: A Flower / PyTorch app."""

import os


import keras
from keras import layers
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
import pandas as pd

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


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

def load_data():
    dados_treino = pd.read_csv("/mnt/fl_conj_treino_cliente.csv")
    dados_teste = pd.read_csv("/mnt/fl_conj_valid_cliente.csv")
    x_train, y_train = dados_treino[['x1', 'x2']], dados_treino['Resposta']
    x_test, y_test = dados_teste[['x1', 'x2']], dados_teste['Resposta']

    return x_train, y_train, x_test, y_test


