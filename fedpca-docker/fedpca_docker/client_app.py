"""quickstart-docker-2: A Flower / PyTorch app."""
import os
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, parameters_to_ndarrays, ndarrays_to_parameters, ParametersRecord, ConfigRecord
from fedpca_docker.task import load_data, load_model, get_model, get_model_params, set_initial_params, set_model_params
from collections import Counter
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.decomposition import TruncatedSVD, PCA
import numpy as np
import pandas as pd
import json

from sklearn.linear_model import LogisticRegression
import warnings

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(
        self, model, data, epochs, batch_size, verbose, context: Context, metodo, penalty, logreg
    ):
        # self.client_state = context.state
        self.model = model
        self.x_train, self.y_train, self.x_test, self.y_test = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.metodo = metodo
        self.penalty = penalty
        self.logreg = logreg

    def fit(self, parameters, config):
        current_round = config['current_round']

        if current_round == 1 and not(os.path.exists("/app/x_train_pca.csv")):
           num_examples = len(self.x_train)
           local_sum = np.sum(self.x_train, axis = 0)
           local_sum_squares = np.sum(self.x_train**2, axis = 0)
           return [], num_examples, {'local_sum': json.dumps(local_sum.tolist()), 'local_sum_squares': json.dumps(local_sum_squares.tolist()), 'pca_exist': os.path.exists("/app/x_train_pca.csv")}

        elif current_round == 2 and not(os.path.exists("/app/x_train_pca.csv")):
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
           return [], num_examples, {'local_sv': sv, 'local_rsv': rsv, 'pca_exist': os.path.exists("/app/x_train_pca.csv")}

        elif current_round == 3 and not(os.path.exists("/app/x_train_pca.csv")):
           ap_global_sv = np.array(json.loads(config['ap_global_sv']))
           ap_global_rsv = np.array(json.loads(config['ap_global_rsv']))
           self.x_train = self.x_train @ ap_global_rsv.T
           self.x_test = self.x_test @ ap_global_rsv.T
           classes = np.array([0, 1, 2, 3])
           counts = list(Counter(self.y_train).values())
           weights = sum(counts)/(counts*len(classes))

           if self.metodo == 'Rede Neural': 
              self.model.set_weights(parameters)
              self.model.fit(
               self.x_train,
               self.y_train,
               epochs=self.epochs,
               batch_size=self.batch_size,
               verbose=self.verbose, 
               class_weight = dict(zip(classes, weights))
              )
              model_parameters = self.model.get_weights()
              y_pred = self.model.predict(self.x_train, verbose = 0)
              y_pred = np.argmax(y_pred, axis = 1)

           elif self.metodo == 'Regressão Logística': 
              set_model_params(self.logreg, parameters)
              # Ignore convergence failure due to low local epochs
              with warnings.catch_warnings():
                  warnings.simplefilter("ignore")
                  self.logreg.fit(self.x_train, self.y_train)
              model_parameters = get_model_params(self.logreg)
              y_pred = self.logreg.predict(self.x_train)
         
           conf_matrix = confusion_matrix(self.y_train, y_pred, labels = [0, 1, 2, 3])
           conf_matrix = json.dumps(conf_matrix.tolist())
           np.savetxt("/app/x_train_pca.csv", self.x_train, delimiter = ',')
           np.savetxt("/app/x_test_pca.csv", self.x_test, delimiter = ',')
           return model_parameters, len(self.x_train), {'pca_exist': os.path.exists("/app/x_train_pca.csv"), 'conf_matrix': conf_matrix, 'metodo': self.metodo}

        else: 
           if self.metodo == 'Rede Neural':
              self.model.set_weights(parameters)
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
              set_model_params(self.logreg, parameters)
              # Ignore convergence failure due to low local epochs
              with warnings.catch_warnings():
                  warnings.simplefilter("ignore")
                  self.logreg.fit(self.x_train, self.y_train)
              model_parameters = get_model_params(self.logreg)
              y_pred = self.logreg.predict(self.x_train)

           conf_matrix = confusion_matrix(self.y_train, y_pred, labels = [0, 1, 2, 3])
           conf_matrix = json.dumps(conf_matrix.tolist())
           return model_parameters, len(self.x_train), {'pca_exist': os.path.exists("/app/x_train_pca.csv"), 'conf_matrix': conf_matrix, 'metodo': self.metodo}
           #return [], len(self.x_train), {'pca_exist': os.path.exists("/app/x_train_pca.csv"), 'metodo': self.metodo}
    def evaluate(self, parameters, config):

           if self.metodo == 'Rede Neural':
              self.model.set_weights(parameters)
              loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
              y_pred = self.model.predict(self.x_test, verbose = 0)
              y_pred = np.argmax(y_pred, axis = 1)

           elif self.metodo == 'Regressão Logística':
              set_model_params(self.logreg, parameters)
              loss = log_loss(self.y_test, self.logreg.predict_proba(self.x_test))
              accuracy = self.logreg.score(self.x_test, self.y_test)
              y_pred = self.logreg.predict(self.x_test)

           conf_matrix = confusion_matrix(self.y_test, y_pred, labels = [0, 1, 2, 3])
           conf_matrix = json.dumps(conf_matrix.tolist())
           return loss, len(self.x_test), {'accuracy': accuracy, 'conf_matrix': conf_matrix}

def client_fn(context: Context):
    # Load model and data
    net = load_model()
    data = load_data()
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config["verbose"]
    metodo = context.run_config['metodo']
    penalty = context.run_config['penalty']
    logreg = get_model(penalty, epochs)
   
    # Setting initial parameters, akin to model.compile for keras models
    set_initial_params(logreg)

    # Return Client instance
    return FlowerClient(
        net, data, epochs, batch_size, verbose, context, metodo, penalty, logreg,
    ).to_client()

# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)

