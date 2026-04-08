
"""neural-sim-ic: A Flower / TensorFlow app."""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = '0'
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

from tensorflow.keras import layers, initializers
import pandas as pd



def load_model():
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(2,)),
            layers.Dense(200, activation='relu', kernel_initializer = initializers.GlorotUniform(seed=1), bias_initializer = 'zeros'),
            #layers.Dense(200, activation='relu', kernel_initializer = initializers.GlorotUniform(seed=1), bias_initializer = 'zeros'),
            layers.Dense(1, activation="sigmoid", kernel_initializer = initializers.GlorotUniform(seed=1), bias_initializer = 'zeros'),
        ]
    )
    model.compile("SGD", "binary_crossentropy", metrics=["accuracy"])
    return model

def load_data(partition_id, num_partitions):
    #dados_treino = pd.read_csv("~/ic_erick/dados_simulacao/IC_S_fl_conj_treino_total.csv")
    dados_teste = pd.read_csv("~/ic_erick/dados_simulacao/IC_S_fl_conj_valid_total.csv")
    # Dados para retreino após seleção de modelo
    dados_treino = pd.read_csv("~/ic_erick/dados_simulacao/IC_S_fl_conj_treino_final.csv")
    dataset_treino = dados_treino[dados_treino['Cliente'] == (partition_id + 1)]
    dataset_teste = dados_teste[dados_teste['Cliente'] == (partition_id + 1)]
    x_train, y_train = dataset_treino[['x1', 'x2']], dataset_treino['Resposta']
    x_test, y_test = dataset_teste[['x1', 'x2']], dataset_teste['Resposta']
    return x_train, y_train, x_test, y_test
