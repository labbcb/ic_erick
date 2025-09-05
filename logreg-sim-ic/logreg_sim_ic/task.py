"""logreg-sim-ic: A Flower / sklearn app."""
# Reprodutibilidade
import random
random.seed(1)
import numpy as np
np.random.seed(1)
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.experimental.enable_op_determinism()
tf.random.set_seed(1)

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from sklearn.linear_model import LogisticRegression
import pandas as pd

fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int):
    """Load partition MNIST data."""
    dados_treino = pd.read_csv("~/ic_erick/dados_simulacao/IC_S_fl_conj_treino_total.csv")
    dados_teste = pd.read_csv("~/ic_erick/dados_simulacao/IC_S_fl_conj_valid_total.csv")
    # Junção dos dados para treino após seleção de hiperparâmetros
    dados_treino = pd.concat([dados_treino, dados_teste])
    dataset_treino = dados_treino[dados_treino['Cliente'] == (partition_id + 1)]
    dataset_teste = dados_teste[dados_teste['Cliente'] == (partition_id + 1)]
    X_train, y_train = dataset_treino[['x1', 'x2']], dataset_treino['Resposta']
    X_test, y_test = dataset_teste[['x1', 'x2']], dataset_teste['Resposta']

    return X_train, X_test, y_train, y_test


def get_model(penalty: str, local_epochs: int):

    return LogisticRegression(
        penalty=penalty,
        max_iter=local_epochs,
        warm_start=True,
        random_state=1
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
    n_classes = 2  
    n_features = 2  # Number of features in dataset
    model.classes_ = np.array([i for i in range(2)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))
