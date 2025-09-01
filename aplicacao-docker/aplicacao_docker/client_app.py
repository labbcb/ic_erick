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
        if self.context.run_config['algoritmo'] == 'Autoencoder':
           self.autoencoder = load_autoencoder_model(input_size = self.context.run_config['input_size'], encoded_size = self.context.run_config['encoded_size'])

    def fit(self, parameters, config):
        current_round = config['current_round']

        if self.context.run_config['algoritmo'] == 'AP-COV':
           if current_round == 1:
              num_examples = len(self.x_train)
              local_sum = np.sum(self.x_train, axis = 0)
              local_sum_squares = np.sum(self.x_train**2, axis = 0)
              return [], num_examples, {'local_sum': json.dumps(local_sum.tolist()), 'local_sum_squares': json.dumps(local_sum_squares.tolist())}
           elif current_round == 2: 
              num_examples = len(self.x_train)
              global_mean = np.array(json.loads(config['global_mean']))
              global_std = np.array(json.loads(config['global_std']))
              self.x_train = (self.x_train - global_mean) / global_std
              self.x_test = (self.x_test - global_mean) / global_std
              k = int(config['k_components'])
              tsvd = TruncatedSVD(n_components = k-1, algorithm = 'arpack')
              tsvd.fit(np.array(self.x_train))
              sv = json.dumps(tsvd.singular_values_.tolist())
              rsv = json.dumps(tsvd.components_.tolist())
              return [], num_examples, {'local_sv': sv, 'local_rsv': rsv}
           elif current_round == 3:
              ap_global_sv = np.array(json.loads(config['ap_global_sv']))
              ap_global_rsv = np.array(json.loads(config['ap_global_rsv']))
              self.x_train = self.x_train @ ap_global_rsv.T
              self.x_test = self.x_test @ ap_global_rsv.T
              np.savetxt("/app/x_train_pca.csv", self.x_train, delimiter = ',')
              np.savetxt("/app/x_valid_pca.csv", self.x_test, delimiter = ',')
              if self.context.run_config['oversample'] == True:
                 counts = list(Counter(self.y_train).values())
                 if min(counts) >=2:
                    smote = SMOTE(random_state = 1, k_neighbors = min(5, min(counts)-1))
                    x_resampled, y_resampled = smote.fit_resample(X = np.array(self.x_train), y = np.array(self.y_train))
                    self.x_train, self.y_train = x_resampled, y_resampled
                    np.savetxt("/app/x_train_pca_resampled.csv", x_resampled, delimiter = ',')
                    np.savetxt("/app/y_train_pca_resampled.csv", y_resampled, delimiter = ',')
              return [], len(self.x_train), {}

        elif self.context.run_config['algoritmo'] == 'Autoencoder':
           if current_round == 1:
              num_examples = len(self.x_train)
              local_sum = np.sum(self.x_train, axis = 0)
              local_sum_squares = np.sum(self.x_train**2, axis = 0)
              return [], num_examples, {'local_sum': json.dumps(local_sum.tolist()), 'local_sum_squares': json.dumps(local_sum_squares.tolist())}

           elif current_round == 2:
              self.autoencoder.fit(
               self.x_train,
               self.x_train,
               epochs = self.context.run_config['ac_epochs'],
               batch_size = self.context.run_config['ac_batch_size'],
               verbose = 1,
              )
              model_parameters = self.autoencoder.get_weights()
              ac_loss_train = self.autoencoder.evaluate(self.x_train, self.x_train, verbose = 0)
              ac_loss_test = self.autoencoder.evaluate(self.x_test, self.x_test, verbose = 0)
              return model_parameters, len(self.x_train), {'ac_loss_train': ac_loss_train, 'ac_loss_test': ac_loss_test, 'n_test': len(self.x_test)}

           elif 3 <= current_round <= self.context.run_config['num-server-rounds'] - 1:
              self.autoencoder.set_weights(parameters)
              self.autoencoder.fit(
               self.x_train,
               self.x_train,
               epochs = self.context.run_config['ac_epochs'],
               batch_size = self.context.run_config['ac_batch_size'],
               verbose = 1,
              )
              model_parameters = self.autoencoder.get_weights()
              ac_loss_train = self.autoencoder.evaluate(self.x_train, self.x_train, verbose = 0)
              ac_loss_test = self.autoencoder.evaluate(self.x_test, self.x_test, verbose = 0)
              return model_parameters, len(self.x_train), {'ac_loss_train': ac_loss_train, 'ac_loss_test': ac_loss_test, 'n_test': len(self.x_test)}

           elif current_round == self.context.run_config['num-server-rounds']:
              self.autoencoder.set_weights(parameters)
              input_layer = keras.Input(shape = (60660,))
              bn_layer = self.autoencoder.get_layer(name = 'bn')(input_layer)
              encoder = keras.Model(input_layer, bn_layer)
              x_train_ac, x_test_ac = encoder.predict(self.x_train), encoder.predict(self.x_test)
              np.savetxt("/app/x_train_ac.csv", x_train_ac, delimiter = ',')
              np.savetxt("/app/x_valid_ac.csv", x_test_ac, delimiter = ',')
              self.x_train, self.x_test = x_train_ac, x_test_ac
              print(self.x_train.shape)
              if self.context.run_config['oversample'] == True:
                 counts = list(Counter(self.y_train).values())
                 if min(counts) >= 2:
                    smote = SMOTE(random_state = 1, k_neighbors = min(5, min(counts)-1), sampling_strategy = 'minority')
                    x_resampled, y_resampled = smote.fit_resample(X = np.array(self.x_train), y = np.array(self.y_train))
                    self.x_train, self.y_train = x_resampled, y_resampled
                    print(self.x_train.shape)
                    np.savetxt("/app/x_train_ac_resampled.csv", x_resampled, delimiter = ',')
                    np.savetxt("/app/y_train_ac_resampled.csv", y_resampled, delimiter = ',')
              return [], len(self.x_train), {}

        elif self.context.run_config['algoritmo'] == 'Rede Neural':
           self.model.set_weights(parameters)
           self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.context.run_config['epochs'],
            batch_size=self.context.run_config['batch_size'],
            verbose=1,
           )
           model_parameters = self.model.get_weights()
           loss_train, accuracy_train = self.model.evaluate(self.x_train, self.y_train, verbose=0)
           loss_test, accuracy_test = self.model.evaluate(self.x_test, self.y_test, verbose=0)
           y_pred_train = self.model.predict(self.x_train, verbose = 0)
           y_pred_train = np.argmax(y_pred_train, axis = 1)
           conf_matrix_train = confusion_matrix(self.y_train, y_pred_train, labels = [0, 1, 2, 3])
           conf_matrix_train = json.dumps(conf_matrix_train.tolist())
           y_pred_test = self.model.predict(self.x_test, verbose = 0)
           y_pred_test = np.argmax(y_pred_test, axis = 1)
           conf_matrix_test = confusion_matrix(self.y_test, y_pred_test, labels = [0, 1, 2, 3])
           conf_matrix_test = json.dumps(conf_matrix_test.tolist())
           return model_parameters, len(self.x_train), {'accuracy_train': accuracy_train, 'accuracy_test': accuracy_test, 
                                                        'conf_matrix_train': conf_matrix_train, 'conf_matrix_test': conf_matrix_test, 
                                                        'loss_train': loss_train, 'loss_test': loss_test,
                                                        'n_test': len(self.x_test)}

        elif self.context.run_config['algoritmo'] == 'Regressão Logística':
           set_model_params(self.model, parameters)
           # Ignore convergence failure due to low local epochs
           with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(self.x_train, self.y_train)
           model_parameters = get_model_params(self.model)
           accuracy_train = self.model.score(self.x_train, self.y_train)
           accuracy_test = self.model.score(self.x_test, self.y_test)
           loss_train = log_loss(self.y_train, self.model.predict_proba(self.x_train), labels = [0, 1, 2, 3])
           loss_test = log_loss(self.y_test, self.model.predict_proba(self.x_test), labels = [0, 1, 2, 3])
           y_pred_train = self.model.predict(self.x_train)
           conf_matrix_train = confusion_matrix(self.y_train, y_pred_train, labels = [0, 1, 2, 3])
           conf_matrix_train = json.dumps(conf_matrix_train.tolist())
           y_pred_test = self.model.predict(self.x_test)
           conf_matrix_test = confusion_matrix(self.y_test, y_pred_test, labels = [0, 1, 2, 3])
           conf_matrix_test = json.dumps(conf_matrix_test.tolist())
           return model_parameters, len(self.x_train), {'accuracy_train': accuracy_train, 'accuracy_test': accuracy_test, 
                                                        'conf_matrix_train': conf_matrix_train, 'conf_matrix_test': conf_matrix_test, 
                                                        'loss_train': loss_train, 'loss_test': loss_test,
                                                        'n_test': len(self.x_test)}

    def evaluate(self, parameters, config):
        current_round = config['current_round']

        if self.context.run_config['algoritmo'] == 'AP-COV':
           return -1.0, len(self.x_test), {}

        elif self.context.run_config['algoritmo'] == 'Autoencoder':
           if current_round == 1:
              return -1.0, len(self.x_test), {}
           elif 2 <= current_round <= self.context.run_config['num-server-rounds'] - 1:
              self.autoencoder.set_weights(parameters)
              ac_loss_train = self.autoencoder.evaluate(self.x_train, self.x_train, verbose = 0)
              ac_loss_test = self.autoencoder.evaluate(self.x_test, self.x_test, verbose = 0)
              return -1.0, len(self.x_test), {'ac_loss_train': ac_loss_train, 'ac_loss_test': ac_loss_test, 'n_train': len(self.x_train)}
           elif current_round == self.context.run_config['num-server-rounds']:
              return -1.0, len(self.x_test), {}

        elif self.context.run_config['algoritmo'] == 'Rede Neural':
              self.model.set_weights(parameters)
              loss_train, accuracy_train = self.model.evaluate(self.x_train, self.y_train, verbose=0)
              loss_test, accuracy_test = self.model.evaluate(self.x_test, self.y_test, verbose=0)
              y_pred_train = self.model.predict(self.x_train, verbose = 0)
              y_pred_train = np.argmax(y_pred_train, axis = 1)
              conf_matrix_train = confusion_matrix(self.y_train, y_pred_train, labels = [0, 1, 2, 3])
              conf_matrix_train = json.dumps(conf_matrix_train.tolist())
              y_pred_test = self.model.predict(self.x_test, verbose = 0)
              y_pred_test = np.argmax(y_pred_test, axis = 1)
              conf_matrix_test = confusion_matrix(self.y_test, y_pred_test, labels = [0, 1, 2, 3])
              conf_matrix_test = json.dumps(conf_matrix_test.tolist())
              return -1.0, len(self.x_test), {'accuracy_train': accuracy_train, 'accuracy_test': accuracy_test,
                                              'conf_matrix_train': conf_matrix_train, 'conf_matrix_test': conf_matrix_test, 
                                              'loss_train': loss_train, 'loss_test': loss_test, 
                                              'n_train': len(self.x_train)}

        elif self.context.run_config['algoritmo'] == 'Regressão Logística':
              set_model_params(self.model, parameters)
              loss_train = log_loss(self.y_train, self.model.predict_proba(self.x_train), labels = [0, 1, 2, 3])
              loss_test = log_loss(self.y_test, self.model.predict_proba(self.x_test), labels = [0, 1, 2, 3])
              accuracy_train = self.model.score(self.x_train, self.y_train)
              accuracy_test = self.model.score(self.x_test, self.y_test)
              y_pred_train = self.model.predict(self.x_train)
              conf_matrix_train = confusion_matrix(self.y_train, y_pred_train, labels = [0, 1, 2, 3])
              conf_matrix_train = json.dumps(conf_matrix_train.tolist())
              y_pred_test = self.model.predict(self.x_test)
              conf_matrix_test = confusion_matrix(self.y_test, y_pred_test, labels = [0, 1, 2, 3])
              conf_matrix_test = json.dumps(conf_matrix_test.tolist())
              return -1.0, len(self.x_test), {'accuracy_train': accuracy_train, 'accuracy_test': accuracy_test,
                                              'conf_matrix_train': conf_matrix_train, 'conf_matrix_test': conf_matrix_test,
                                              'loss_train': loss_train, 'loss_test': loss_test, 
                                              'n_train': len(self.x_train)}

def client_fn(context: Context):
    # Load model and data
    data = load_data(algoritmo = context.run_config['algoritmo'],
                     tipo_dados = context.run_config['tipo_dados'],
                     oversample = context.run_config['oversample'])

    if context.run_config['algoritmo'] == 'Regressão Logística':
       model = get_model(penalty = context.run_config['penalty'],
                         C = context.run_config['C'],
                         solver = context.run_config['solver'],
                         max_iter = context.run_config['max_iter'])
       set_initial_params(model,
                          n_classes = context.run_config['n_classes'],
                          n_variaveis = context.run_config['n_variaveis'])

    elif context.run_config['algoritmo'] == 'Rede Neural':
       model = load_model(n_variaveis = context.run_config['n_variaveis'])

    else:
       model = None

    # Return Client instance
    return FlowerClient(
        model, data, context,
    ).to_client()

# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)

