"""quickstart-docker-2: A Flower / PyTorch app."""
import os
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, parameters_to_ndarrays, ndarrays_to_parameters, ParametersRecord, ConfigRecord
from fedpca_docker.task import load_data, load_model, get_model, get_model_params, set_initial_params, set_model_params, load_autoencoder_model
from collections import Counter
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd
import json
import keras
from imblearn.over_sampling import ADASYN, SMOTE

from sklearn.linear_model import LogisticRegression
import warnings

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(
       #self, model, data, epochs, batch_size, verbose, context: Context, metodo, penalty, logreg, preprocessing
       self, model, data, context
    ):
        self.model = model
        self.x_train, self.y_train, self.x_test, self.y_test = data
        self.context = context
        self.pre_processamento = context.run_config['pre_processamento']
        self.tipo_dados = context.run_config['tipo_dados']
        if self.tipo_dados == 'autoencoder':
           self.input_size = context.run_config['input_size']
           self.encoded_size = context.run_config['encoded_size']
           self.autoencoder = load_autoencoder_model(input_size = self.input_size, encoded_size = self.encoded_size)
           self.ac_epochs = context.run_config['ac_epochs']
           self.ac_batch_size = context.run_config['ac_batch_size']
        #self.client_state = context.state
        self.metodo = context.run_config['metodo']
        self.oversample = context.run_config['oversample']
        self.epochs = context.run_config['epochs']
        self.batch_size = context.run_config['batch_size']
        self.verbose = context.run_config['verbose']

    def fit(self, parameters, config):
        current_round = config['current_round']

        if current_round == 1 and self.context.run_config['pre_processamento'] == True:
           num_examples = len(self.x_train)
           local_sum = np.sum(self.x_train, axis = 0)
           local_sum_squares = np.sum(self.x_train**2, axis = 0)
           return [], num_examples, {'local_sum': json.dumps(local_sum.tolist()), 'local_sum_squares': json.dumps(local_sum_squares.tolist())}

        if current_round == 2 and self.context.run_config['pre_processamento'] == True: 
           num_examples = len(self.x_train)
           global_mean = np.array(json.loads(config['global_mean']))
           global_std = np.array(json.loads(config['global_std']))
           self.x_train = (self.x_train - global_mean) / global_std
           self.x_test = (self.x_test - global_mean) / global_std
           if self.context.run_config['tipo_dados'] == 'fedpca':
              k = int(config['k_components'])
              tsvd = TruncatedSVD(n_components = k-1, algorithm = 'arpack')
              tsvd.fit(np.array(self.x_train))
              sv = json.dumps(tsvd.singular_values_.tolist())
              rsv = json.dumps(tsvd.components_.tolist())
              return [], num_examples, {'local_sv': sv, 'local_rsv': rsv}
           elif self.tipo_dados == 'autoencoder':
              self.autoencoder.fit(
               self.x_train,
               self.x_train,
               epochs = self.context.run_config['ac_epochs'],
               batch_size = self.context.run_config['ac_batch_size'],
               verbose = 1,
              )
              model_parameters = self.autoencoder.get_weights()
           return model_parameters, len(self.x_train), {}

        if current_round == 3 and self.pre_processamento == True:
           if self.context.run_config['tipo_dados'] == 'fedpca':
              ap_global_sv = np.array(json.loads(config['ap_global_sv']))
              ap_global_rsv = np.array(json.loads(config['ap_global_rsv']))
              self.x_train = self.x_train @ ap_global_rsv.T
              self.x_test = self.x_test @ ap_global_rsv.T
              np.savetxt("/app/x_train_pca.csv", self.x_train, delimiter = ',')
              np.savetxt("/app/x_valid_pca.csv", self.x_test, delimiter = ',')
              classes = [0, 1, 2, 3]
              if self.oversample == True:
                 counts = list(Counter(self.y_train).values())
                 if min(counts) >=2:
                    smote = SMOTE(random_state = 1, k_neighbors = min(5, min(counts)-1))
                    x_resampled, y_resampled = smote.fit_resample(X = np.array(self.x_train), y = np.array(self.y_train))
                    self.x_train, self.y_train = x_resampled, y_resampled
                    np.savetxt("/app/x_train_pca_resampled.csv", x_resampled, delimiter = ',')
                    np.savetxt("/app/y_train_pca_resampled.csv", y_resampled, delimiter = ',')
                 self.oversample = False
              print(self.x_train.shape)
              if self.metodo == 'Rede Neural': 
                 self.model.fit(
                  self.x_train,
                  self.y_train,
                  epochs=self.context.run_config['epochs'],
                  batch_size=self.context.run_config['batch_size'],
                  verbose=1,
                 )
                 model_parameters = self.model.get_weights()
                 y_pred = self.model.predict(self.x_train, verbose = 0)
                 y_pred = np.argmax(y_pred, axis = 1)
              elif self.metodo == 'Regressão Logística':
                 # Ignore convergence failure due to low local epochs
                 with warnings.catch_warnings():
                     warnings.simplefilter("ignore")
                     self.model.fit(self.x_train, self.y_train)
                 model_parameters = get_model_params(self.model)
                 y_pred = self.model.predict(self.x_train)
              conf_matrix = confusion_matrix(self.y_train, y_pred, labels = [0, 1, 2, 3])
              conf_matrix = json.dumps(conf_matrix.tolist())
              return model_parameters, len(self.x_train), {'conf_matrix': conf_matrix}
           elif self.tipo_dados == 'autoencoder':
              self.autoencoder.set_weights(parameters)
              self.autoencoder.fit(
               self.x_train,
               self.x_train,
               epochs = self.context.run_config['ac_epochs'],
               batch_size = self.context.run_config['ac_batch_size'],
               verbose = 1,
              )
              model_parameters = self.autoencoder.get_weights()
              return model_parameters, len(self.x_train), {}

        #elif 4 <= current_round <=7 and (self.preprocessing == 'autoencoder' and not(os.path.exists("/app/x_train_ac.csv"))):
        elif 4 <= current_round <= 6 and self.pre_processamento == True and self.tipo_dados == 'autoencoder': 
           self.autoencoder.set_weights(parameters)
           self.autoencoder.fit(
            self.x_train,
            self.x_train,
            epochs=self.context.run_config['ac_epochs'],
            batch_size=self.context.run_config['ac_batch_size'],
            verbose=1,
           )
           model_parameters = self.autoencoder.get_weights()
           return model_parameters, len(self.x_train), {}

        elif current_round == 7  and self.pre_processamento == True and self.tipo_dados == 'autoencoder':
           self.autoencoder.set_weights(parameters)
           input_layer = keras.Input(shape = (60660,))
           bn_layer = self.autoencoder.get_layer(name = 'bn')(input_layer)
           encoder = keras.Model(input_layer, bn_layer)
           x_train_ac, x_test_ac = encoder.predict(self.x_train), encoder.predict(self.x_test)
           np.savetxt("/app/x_train_ac.csv", x_train_ac, delimiter = ',')
           np.savetxt("/app/x_valid_ac.csv", x_test_ac, delimiter = ',')
           self.x_train, self.x_test = x_train_ac, x_test_ac
           print(self.x_train.shape)

           if self.oversample == True:
                 counts = list(Counter(self.y_train).values())
                 if min(counts) >= 2:
                    smote = SMOTE(random_state = 1, k_neighbors = min(5, min(counts)-1), sampling_strategy = 'minority')
                    x_resampled, y_resampled = smote.fit_resample(X = np.array(self.x_train), y = np.array(self.y_train))
                    self.x_train, self.y_train = x_resampled, y_resampled
                    print(self.x_train.shape)
                    np.savetxt("/app/x_train_ac_resampled.csv", x_resampled, delimiter = ',')
                    np.savetxt("/app/y_train_ac_resampled.csv", y_resampled, delimiter = ',')
                 self.oversample = False
           if self.metodo == 'Rede Neural':
              #self.model.set_weights(parameters)
              counts = list(Counter(self.y_train).values())
              self.model.fit(
               self.x_train,
               self.y_train,
               epochs=self.epochs,
               batch_size=self.batch_size,
               verbose=self.verbose,
              )
              model_parameters = self.model.get_weights()
              y_pred = self.model.predict(self.x_train, verbose = 0)
              y_pred = np.argmax(y_pred, axis = 1)

           elif self.metodo == 'Regressão Logística':
              # Ignore convergence failure due to low local epochs
              with warnings.catch_warnings():
                  warnings.simplefilter("ignore")
                  self.model.fit(self.x_train, self.y_train)
              model_parameters = get_model_params(self.model)
              y_pred = self.model.predict(self.x_train)
           conf_matrix = confusion_matrix(self.y_train, y_pred, labels = [0, 1, 2, 3])
           conf_matrix = json.dumps(conf_matrix.tolist())
           return model_parameters, len(self.x_train), {'conf_matrix': conf_matrix}

        else:
           print(self.x_train.shape)
           if self.oversample == True:
              counts = list(Counter(self.y_train).values())
              if min(counts) >= 2:
                 print(counts)
                 smote = SMOTE(random_state = 1, k_neighbors = min(5, min(counts)-1))
                 x_resampled, y_resampled = smote.fit_resample(X = np.array(self.x_train), y = np.array(self.y_train))
                 self.x_train, self.y_train = x_resampled, y_resampled
                 np.savetxt("/app/x_train_ac_resampled.csv", x_resampled, delimiter = ',')
                 np.savetxt("/app/y_train_ac_resampled.csv", y_resampled, delimiter = ',')
              self.oversample = False

           if self.metodo == 'Rede Neural':
              self.model.set_weights(parameters)
              self.model.fit(
               self.x_train,
               self.y_train,
               epochs=self.context.run_config['epochs'],
               batch_size=self.context.run_config['batch_size'],
               verbose=1,
              )
              model_parameters = self.model.get_weights()
              y_pred = self.model.predict(self.x_train, verbose = 0)
              y_pred = np.argmax(y_pred, axis = 1)

           elif self.metodo == 'Regressão Logística':
              set_model_params(self.model, parameters)
              # Ignore convergence failure due to low local epochs
              with warnings.catch_warnings():
                  warnings.simplefilter("ignore")
                  self.model.fit(self.x_train, self.y_train)
              model_parameters = get_model_params(self.model)
              y_pred = self.model.predict(self.x_train)

           conf_matrix = confusion_matrix(self.y_train, y_pred, labels = [0, 1, 2, 3])
           conf_matrix = json.dumps(conf_matrix.tolist())
           return model_parameters, len(self.x_train), {'conf_matrix': conf_matrix}

    def evaluate(self, parameters, config):
        current_round = config['current_round']
        if (current_round == 1 and self.context.run_config['pre_processamento'] == True) or (current_round == 2 and self.context.run_config['pre_processamento'] == True and self.context.run_config['tipo_dados'] == 'fedpca'):
           return -1.0, len(self.x_test), {}
        elif 2 <= current_round <= 6  and self.context.run_config['pre_processamento'] == True and self.context.run_config['tipo_dados'] == 'autoencoder':
           self.autoencoder.set_weights(parameters)
           ac_loss = self.autoencoder.evaluate(self.x_test, self.x_test, verbose = 0)
           return -1.0, len(self.x_test), {'ac_loss': ac_loss}
        #elif current_round == 7 and self.context.run_config['pre_processamento'] == True and self.context.run_config['tipo_dados'] == 'autoencoder':
        #   self.autoencoder.set_weights(parameters)
        #   ac_loss = self.autoencoder.evaluate(self.x_test, self.x_test, verbose = 0)
        #   if self.context.run_config['metodo'] == 'Rede Neural':
        #      self.model.set_weights(parameters)
        #      loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        #      y_pred = self.model.predict(self.x_test, verbose = 0)
        #      y_pred = np.argmax(y_pred, axis = 1)
        #   elif self.context.run_config['metodo'] == 'Regressão Logística':
        #      set_model_params(self.model, parameters)
        #      loss = log_loss(self.y_test, self.model.predict_proba(self.x_test))
        #      accuracy = self.model.score(self.x_test, self.y_test)
        #      y_pred = self.model.predict(self.x_test)
        #   conf_matrix = confusion_matrix(self.y_test, y_pred, labels = [0, 1, 2, 3])
        #   conf_matrix = json.dumps(conf_matrix.tolist())
        #   return loss, len(self.x_test), {'accuracy': accuracy, 'conf_matrix': conf_matrix, 'ac_loss': ac_loss}
        else:
           if self.context.run_config['metodo'] == 'Rede Neural':
              self.x_train, self.y_train, self.x_test, self.y_test = load_data(tipo_dados = self.context.run_config['tipo_dados'],
                     pre_processamento = False,
                     oversample = False)
              print(self.x_test.shape)
              self.model.set_weights(parameters)
              loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
              y_pred = self.model.predict(self.x_test, verbose = 0)
              y_pred = np.argmax(y_pred, axis = 1)
           elif self.context.run_config['metodo'] == 'Regressão Logística':
              print(len(parameters[0][0]))
              print(parameters)
              set_model_params(self.model, parameters)
              loss = log_loss(self.y_test, self.model.predict_proba(self.x_test), labels = [0, 1, 2, 3])
              accuracy = self.model.score(self.x_test, self.y_test)
              y_pred = self.model.predict(self.x_test)

           conf_matrix = confusion_matrix(self.y_test, y_pred, labels = [0, 1, 2, 3])
           conf_matrix = json.dumps(conf_matrix.tolist())
           return loss, len(self.x_test), {'accuracy': accuracy, 'conf_matrix': conf_matrix}

def client_fn(context: Context):
    # Load model and data
    data = load_data(tipo_dados = context.run_config['tipo_dados'],
                     pre_processamento = context.run_config['pre_processamento'],
                     oversample = context.run_config['oversample'])

    if context.run_config['metodo'] == 'Regressão Logística':
       model = get_model(penalty = context.run_config['penalty'],
                         C = context.run_config['C'],
                         solver = context.run_config['solver'],
                         max_iter = context.run_config['max_iter'])
       set_initial_params(model,
                          n_classes = context.run_config['n_classes'],
                          n_variaveis = context.run_config['n_variaveis'])

    elif context.run_config['metodo'] == 'Rede Neural':
       model = load_model(n_variaveis = context.run_config['n_variaveis'])

    # Return Client instance
    return FlowerClient(
        model, data, context,
    ).to_client()

# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)

