"""quickstart-docker-logreg: A Flower / PyTorch app."""
"""logreg-sim-ic: A Flower / sklearn app."""

import numpy as np
#from flwr_datasets import FederatedDataset
#from flwr_datasets.partitioner import IidPartitioner
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import LabelEncoder
fds = None  # Cache FederatedDataset


#def load_data(partition_id: int, num_partitions: int):
#def load_data():
    # Only initialize `FederatedDataset` once
#    global fds
#    if fds is None:
#       #dados_treino = load_dataset('csv', data_files = "/mnt/fl_conj_treino_cliente.csv")
#       #dados_teste = load_dataset('csv', data_files = "/mnt/fl_conj_valid_cliente.csv")
#       dados_treino = pd.read_csv("/mnt/fl_conj_treino_cliente.csv")
#       dados_teste = pd.read_csv("/mnt/fl_conj_valid_cliente.csv")
#    X_train, y_train = dados_treino[['x1', 'x2']], dados_treino['Resposta']
#    X_test, y_test = dados_teste[['x1', 'x2']], dados_teste['Resposta']

#    return X_train, X_test, y_train, y_test

def load_data():
    dados_treino = pd.read_csv("/mnt/fl_conj_treino_cliente.csv")
    dados_teste = pd.read_csv("/mnt/fl_conj_teste_cliente.csv")
    X_train, y_train = dados_treino.iloc[:,5:], dados_treino['subtipo2']
    X_test, y_test = dados_teste.iloc[:,5:], dados_teste['subtipo2']
    ee = LabelEncoder()
    y_train = ee.fit_transform(y_train)
    y_test = ee.transform(y_test)

    return X_train, X_test, y_train, y_test

def get_model(penalty: str, local_epochs: int):

    return LogisticRegression(
        penalty=penalty,
        max_iter=local_epochs,
        warm_start=True,
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
    n_features = 60660
    model.classes_ = np.array([i for i in range(4)])
    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))





