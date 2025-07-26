"""quickstart-docker-2: A Flower / PyTorch app."""
import os
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from fedpca_docker.task import load_data, load_model
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import json

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(
        self, model, data, epochs, batch_size, verbose
    ):
        self.model = model
        self.x_train, self.y_train, self.x_test, self.y_test = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, parameters, config):
        current_round = config['current_round']

        if current_round == 1:
           num_examples = len(self.x_train)
           local_sum = json.dumps(list(np.sum(self.x_train, axis = 0)))
           local_sum_squares = json.dumps(list(np.sum(self.x_train**2, axis = 0)))
           return [], num_examples, {'local_sum': local_sum, 'local_sum_squares': local_sum_squares}

        elif current_round == 2:
           global_mean = np.array(json.loads(config['global_mean']))
           global_std = np.array(json.loads(config['global_std']))
           self.x_train = (self.x_train - global_mean) / global_std
           self.x_test = (self.x_test - global_mean) / global_std
           self.model.set_weights(parameters)
           self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
           )
           return self.model.get_weights(), len(self.x_train), {}

        else: 
           self.model.set_weights(parameters)
           self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
           )
           return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        current_round = config['current_round']
        
        if current_round == 1:
           return None

        else: 
           self.model.set_weights(parameters)
           loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
           # y_pred = self.model.predict(self.x_test, verbose = 0)
           # y_pred = np.argmax(y_pred, axis = 1)
           # conf_matrix = confusion_matrix(self.y_test, y_pred, labels = ['0', '1', '2', '3'])
           # conf_matrix = pd.DataFrame(conf_matrix, index = ['0', '1', '2', '3'], columns = ['0', '1', '2', '3'])
           # return loss, len(self.x_test), {"00": int(conf_matrix.iat[0,0]), "01": int(conf_matrix.iat[0,1]), "02": int(conf_matrix.iat[0,2]), "03": int(conf_matrix.iat[0,3]), "10": int(conf_matrix.iat[1,0]), "11": int(conf_matrix.iat[1,1]), "12": int(conf_matrix.iat[1,2]), "13": int(conf_matrix.iat[1,3]),"20": int(conf_matrix.iat[2,0]), "21": int(conf_matrix.iat[2,1]), "22": int(conf_matrix.iat[2,2]), "23": int(conf_matrix.iat[2,3]), "30": int(conf_matrix.iat[3,0]),"31": int(conf_matrix.iat[3,1]), "32": int(conf_matrix.iat[3,2]), "33": int(conf_matrix.iat[3,3])}
           return loss, len(self.x_test), {'accuracy': accuracy}
           # return loss, {"conf_matrix": conf_matrix}

def client_fn(context: Context):
    # Load model and data
    net = load_model()
    data = load_data()
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config["verbose"]

    # Return Client instance
    return FlowerClient(
        net, data, epochs, batch_size, verbose
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)

