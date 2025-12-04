
"""quickstart-docker-2: A Flower / PyTorch app."""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
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

from tensorflow import keras
from keras import layers, regularizers, initializers, Sequential, Input, optimizers
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

'''
def load_model(n_variaveis: int):
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(n_variaveis,)),
            #layers.Dense(10, activation='relu'),
            #layers.Dense(10, activation='relu'),
            layers.Dense(20, activation='relu', kernel_regularizer = regularizers.L1(1e-4), kernel_initializer = initializers.GlorotUniform(seed=1), bias_initializer = 'zeros'),
            layers.Dense(20, activation='relu', kernel_regularizer = regularizers.L1(1e-4), kernel_initializer = initializers.GlorotUniform(seed=1), bias_initializer = 'zeros'),
            layers.Dense(4, activation="softmax", kernel_initializer = initializers.GlorotUniform(seed=1)),
        ]
    )
    model.compile(optimizers.SGD(learning_rate = 5e-4), "sparse_categorical_crossentropy", metrics=['accuracy'])
    return(model)
'''
def load_model(n_variaveis: int, hidden_layer_size, hidden_layer_num, regularizer, n_classes):
    model = Sequential()
    model.add(Input(shape=(n_variaveis,)))
    for i in range(hidden_layer_num):
        model.add(
            layers.Dense(
                units=hidden_layer_size,
                activation='relu',
                kernel_regularizer = regularizer,
                kernel_initializer = initializers.GlorotUniform(seed=1),
                bias_initializer = 'zeros'
        )
    )
    model.add(layers.Dense(n_classes, activation='softmax',
                                 kernel_initializer = initializers.GlorotUniform(seed=1),
                                 bias_initializer = 'zeros'))
    model.compile(
         optimizer = keras.optimizers.SGD(learning_rate=5e-4),
         loss='sparse_categorical_crossentropy',
         metrics=["accuracy"]
    )
    return model

def load_autoencoder_model(input_size: int, encoded_size: int): 
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(input_size,), name = 'input'),
            layers.Dense(encoded_size, activation='relu', name = 'bn', kernel_initializer = initializers.GlorotUniform(seed=1), bias_initializer = 'zeros'),
            layers.Dense(input_size, activation="sigmoid", name = 'output', kernel_initializer = initializers.GlorotUniform(seed=1), bias_initializer = 'zeros'),
        ]
    )
    model.compile(optimizers.SGD(learning_rate = 5e-4), loss = 'mse')
    return(model)

def load_data(algoritmo, tipo_dados, oversample):
    if algoritmo == 'Rede Neural' or algoritmo == 'Regressão Logística':
       if tipo_dados == 'AP-COV':
          if oversample and os.path.exists("/app/x_train_pca_resampled.csv"):
             x_train = pd.read_csv("/app/x_train_pca_resampled.csv", header = None)
             #x_test = pd.read_csv("/app/x_valid_pca.csv", header = None)
             x_test = pd.read_csv("/app/x_test_pca.csv", header = None)
             y_train = pd.read_csv("/app/y_train_pca_resampled.csv", header = None)
             #y_test = pd.read_csv("/mnt/fl_conj_valid_cliente.csv")['subtipo2']
             y_test = pd.read_csv("/mnt/fl_conj_test_cliente.csv")['subtipo2']
             ee = LabelEncoder()
             ee.fit(['BRCA.Basal', 'BRCA.Her2','BRCA.LumA', 'BRCA.LumB'])
             y_test = ee.transform(y_test)
          else:
             x_train = pd.read_csv("/app/x_train_pca.csv", header = None)
             #x_test = pd.read_csv("/app/x_valid_pca.csv", header = None)
             x_test = pd.read_csv("/app/x_test_pca.csv", header = None)
             #y_train = pd.read_csv("/mnt/fl_conj_treino_cliente.csv")['subtipo2']
             y_train = pd.concat([pd.read_csv("/mnt/fl_conj_treino_cliente.csv")['subtipo2'], pd.read_csv("/mnt/fl_conj_valid_cliente.csv")['subtipo2']])
             #y_test = pd.read_csv("/mnt/fl_conj_valid_cliente.csv")['subtipo2']
             y_test = pd.read_csv("/mnt/fl_conj_test_cliente.csv")['subtipo2']
             ee = LabelEncoder()
             ee.fit(['BRCA.Basal', 'BRCA.Her2','BRCA.LumA', 'BRCA.LumB'])
             y_train = ee.transform(y_train)
             y_test = ee.transform(y_test)
       elif tipo_dados == 'Autoencoder':
          if oversample and os.path.exists("/app/x_train_ac_resampled.csv"):
             x_train = pd.read_csv("/app/x_train_ac_resampled.csv", header = None)
             x_test = pd.read_csv("/app/x_valid_ac.csv", header = None)
             #x_test = pd.read_csv("/app/x_test_ac.csv", header = None)
             y_train = pd.read_csv("/app/y_train_ac_resampled.csv", header = None)
             y_test = pd.read_csv("/mnt/fl_conj_valid_cliente.csv")['subtipo2']
             #y_test = pd.read_csv("/mnt/fl_conj_test_cliente.csv")['subtipo2']
             ee = LabelEncoder()
             ee.fit(['BRCA.Basal', 'BRCA.Her2','BRCA.LumA', 'BRCA.LumB'])
             y_test = ee.transform(y_test)
          else:
             x_train = pd.read_csv("/app/x_train_ac.csv", header = None)
             x_test = pd.read_csv("/app/x_valid_ac.csv", header = None)
             #x_test = pd.read_csv("/app/x_test_ac.csv", header = None)
             y_train = pd.read_csv("/mnt/fl_conj_treino_cliente.csv")['subtipo2']
             #y_train = pd.concat([pd.read_csv("/mnt/fl_conj_treino_cliente.csv")['subtipo2'], pd.read_csv("/mnt/fl_conj_valid_cliente.csv")['subtipo2']])
             y_test = pd.read_csv("/mnt/fl_conj_valid_cliente.csv")['subtipo2']
             #y_test = pd.read_csv("/mnt/fl_conj_test_cliente.csv")['subtipo2'] 
             ee = LabelEncoder()
             ee.fit(['BRCA.Basal', 'BRCA.Her2','BRCA.LumA', 'BRCA.LumB'])
             y_train = ee.transform(y_train)
             y_test = ee.transform(y_test)
       elif tipo_dados == "SUB-IT":
          if oversample and os.path.exists("/app/x_train_subit_resampled.csv"):
             x_train = pd.read_csv("/app/x_train_subit_resampled.csv", header = None)
             #x_test = pd.read_csv("/app/x_valid_subit.csv", header = None)
             x_test = pd.read_csv("/app/x_test_subit.csv", header = None)
             y_train = pd.read_csv("/app/y_train_subit_resampled.csv", header = None)
             #y_test = pd.read_csv("/mnt/fl_conj_valid_cliente.csv")['subtipo2']
             y_test = pd.read_csv("/mnt/fl_conj_test_cliente.csv")['subtipo2']
             ee = LabelEncoder()
             ee.fit(['BRCA.Basal', 'BRCA.Her2','BRCA.LumA', 'BRCA.LumB'])
             y_test = ee.transform(y_test)
          else:
             x_train = pd.read_csv("/app/x_train_subit.csv", header = None)
             #x_test = pd.read_csv("/app/x_valid_subit.csv", header = None)
             x_test = pd.read_csv("/app/x_test_subit.csv", header = None)
             #y_train = pd.read_csv("/mnt/fl_conj_treino_cliente.csv")['subtipo2']
             y_train = pd.concat([pd.read_csv("/mnt/fl_conj_treino_cliente.csv")['subtipo2'], pd.read_csv("/mnt/fl_conj_valid_cliente.csv")['subtipo2']])
             #y_test = pd.read_csv("/mnt/fl_conj_valid_cliente.csv")['subtipo2']
             y_test = pd.read_csv("/mnt/fl_conj_test_cliente.csv")['subtipo2']
             ee = LabelEncoder()
             ee.fit(['BRCA.Basal', 'BRCA.Her2','BRCA.LumA', 'BRCA.LumB'])
             y_train = ee.transform(y_train)
             y_test = ee.transform(y_test) 
    elif algoritmo == 'AP-COV' or algoritmo == 'Autoencoder' or algoritmo == "SUB-IT":
       	  #dados_treino = pd.read_csv("/mnt/fl_conj_treino_cliente.csv")
          dados_treino = pd.concat([pd.read_csv("/mnt/fl_conj_treino_cliente.csv"), pd.read_csv("/mnt/fl_conj_valid_cliente.csv")])
          #dados_teste = pd.read_csv("/mnt/fl_conj_valid_cliente.csv")
          dados_teste = pd.read_csv("/mnt/fl_conj_test_cliente.csv")
          x_train, y_train = dados_treino.iloc[:,5:], dados_treino['subtipo2']
          x_test, y_test = dados_teste.iloc[:,5:], dados_teste['subtipo2']
          ee = LabelEncoder()
          ee.fit(['BRCA.Basal', 'BRCA.Her2','BRCA.LumA', 'BRCA.LumB'])
          y_train = ee.transform(y_train)
          y_test = ee.transform(y_test)
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


def set_initial_params(model, n_classes, n_variaveis):
    n_classes = n_classes
    n_features = n_variaveis
   
    model.classes_ = np.array([i for i in range(n_classes)])
    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))

def generate_random_gaussian(m, k):
    draws = m*k
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(size = draws)
    noise.shape = (m, k)
    return noise

def eigenvector_convergence_checker(current, previous, tolerance=1e-9, required=None):
    '''

    This function checks whether two sets of vectors are assymptotically collinear, up
    to a tolerance of epsilon.
    Args:
        current: The current eigenvector estimate
        previous: The eigenvector estimate from the previous iteration
        tolerance: The error tolerance for eigenvectors to be equal
        required: optional parameter for the number of eigenvectors required to have converged

    Returns: True if the required numbers of eigenvectors have converged to the given precision, False otherwise
                deltas, the current difference between the dot products

    '''
    nr_converged = 0
    col = 0
    converged = False
    deltas = []
    if required is None:
        required = current.shape[1]
    while col < current.shape[1] and not converged:
        # check if the scalar product of the current and the previous eigenvectors
        # is 1, which means the vectors are 'parallel'
        delta = np.abs(np.sum(np.dot(np.transpose(current[:, col]), previous[:, col])))
        deltas.append(delta)
        if delta >= 1 - tolerance:
            nr_converged = nr_converged + 1
        if nr_converged >= required:
            converged = True
        col = col + 1
    return converged, deltas, nr_converged
