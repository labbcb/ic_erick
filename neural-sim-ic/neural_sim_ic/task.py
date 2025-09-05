"""neural-sim-ic: A Flower / TensorFlow app."""

import os

import sklearn
import keras
from keras import layers
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
import pandas as pd

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_model():
    # Define a simple CNN for CIFAR-10 and set Adam optimizer
    keras.utils.set_random_seed(1)
    model = keras.Sequential(
        [
            keras.Input(shape=(2,)),
            layers.Dense(200, activation='relu'),
            layers.Dense(200, activation='relu'),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile("SGD", "binary_crossentropy", metrics=["accuracy"])
    return model


fds = None  # Cache FederatedDataset


def load_data(partition_id, num_partitions):
    # Download and partition dataset
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        # partitioner = IidPartitioner(num_partitions=num_partitions)
        # fds = FederatedDataset(
        #     dataset="uoft-cs/cifar10",
        #     partitioners={"train": partitioner},
        # )
    # partition = fds.load_partition(partition_id, "train")
    # partition.set_format("numpy")

        #dados_treino = pd.read_csv("C:/Users/erick/Downloads/jupyter_arquivos/Dados_Simulados_IC_Treino_FL.csv")
        #dados_teste = pd.read_csv("C:/Users/erick/Downloads/jupyter_arquivos/Dados_Simulados_IC_Validacao_FL.csv")
        dados_treino = pd.read_csv("C:/Users/erick/IC_S_fl_conj_treino_total.csv")
        dados_teste = pd.read_csv("C:/Users/erick/IC_S_fl_conj_treino_total.csv")
    # Junção de datasets para retreino
    dados_treino = pd.concat([dados_treino, dados_teste])
    dataset_treino = dados_treino[dados_treino['Cliente'] == (partition_id + 1)]
    dataset_teste = dados_teste[dados_teste['Cliente'] == (partition_id + 1)]
    x_train, y_train = dataset_treino[['x1', 'x2']], dataset_treino['Resposta']
    x_test, y_test = dataset_teste[['x1', 'x2']], dataset_teste['Resposta']
    # x_train, x_test = X[: int(0.7 * len(X))], X[int(0.7 * len(X)) :]
    # y_train, y_test = y[: int(0.7 * len(y))], y[int(0.7 * len(y)) :]
    # dataset_teste = dados_teste[dados_teste['Cliente'] == (partition_id + 1)]
    # x_train, y_train = dataset_treino[['x1', 'x2']], dataset_treino['Resposta']
    # x_test, y_test = dataset_teste[['x1', 'x2']], dataset_teste['Resposta']
    # Divide data on each node: 80% train, 20% test
    # partition = partition.train_test_split(test_size=0.2)
    # x_train, y_train = partition["train"]["img"] / 255.0, partition["train"]["label"]
    # x_test, y_test = partition["test"]["img"] / 255.0, partition["test"]["label"]
    return x_train, y_train, x_test, y_test
