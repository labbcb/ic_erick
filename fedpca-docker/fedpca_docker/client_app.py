"""quickstart-docker-2: A Flower / PyTorch app."""
import os
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, parameters_to_ndarrays, ndarrays_to_parameters, ParametersRecord, ConfigRecord
from fedpca_docker.task import load_data, load_model
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import TruncatedSVD, PCA
import numpy as np
import pandas as pd
import json

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(
        self, model, data, epochs, batch_size, verbose, context: Context
    ):
        self.client_state = context.state
        self.model = model
        self.x_train, self.y_train, self.x_test, self.y_test = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        # if "dados_transformados" not in self.client_state.parameters_records:
           # self.client_state.parameters_records["dados_transformados"] = ParametersRecord()

    def fit(self, parameters, config):
        current_round = config['current_round']

        if current_round == 1 and not(os.path.exists("/app/x_train_pca.csv")):
           num_examples = len(self.x_train)
           local_sum = np.sum(self.x_train, axis = 0)
           local_sum_squares = np.sum(self.x_train**2, axis = 0)
           # medidas = ndarrays_to_parameters([local_sum, local_sum_squares])
           # return [local_sum, local_sum_squares], num_examples, {}
           return [], num_examples, {'local_sum': json.dumps(local_sum.tolist()), 'local_sum_squares': json.dumps(local_sum_squares.tolist()), 'pca_exist': os.path.exists("/app/x_train_pca.csv")}

        elif current_round == 2 and not(os.path.exists("/app/x_train_pca.csv")):
           num_examples = len(self.x_train)
           global_mean = np.array(json.loads(config['global_mean']))
           global_std = np.array(json.loads(config['global_std']))
           # global_mean, global_std = parameters
           # np.savetxt("/app/mean.csv", global_mean, delimiter = ',')
           # np.savetxt("/app/std.csv", global_std, delimiter = ',')
           self.x_train = (self.x_train - global_mean) / global_std
           self.x_test = (self.x_test - global_mean) / global_std
           # np.savetxt("dados_t", self.x_train, delimiter = ',')
           k = int(config['k_components'])
           tsvd = TruncatedSVD(n_components = k-1, algorithm = 'arpack')
           # pca = PCA(n_components = k-1)
           # pca.fit(self.x_train)
           tsvd.fit(np.array(self.x_train))
           sv = json.dumps(tsvd.singular_values_.tolist())
           rsv = json.dumps(tsvd.components_.tolist())
           # sv = json.dumps(pca.singular_values_.tolist())
           # rsv = json.dumps(pca.components_.tolist())
           # return [tsvd.singular_values_, tsvd.components_], num_examples, {}
           return [], num_examples, {'local_sv': sv, 'local_rsv': rsv, 'pca_exist': os.path.exists("/app/x_train_pca.csv")}
           # return [], num_examples, {}
        elif current_round == 3 and not(os.path.exists("/app/x_train_pca.csv")):
           ap_global_sv = np.array(json.loads(config['ap_global_sv']))
           ap_global_rsv = np.array(json.loads(config['ap_global_rsv']))
           # ap_global_sv, ap_global_rsv = parameters
           self.x_train = self.x_train @ ap_global_rsv.T
           self.x_test = self.x_test @ ap_global_rsv.T
           self.model.set_weights(parameters)
           self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
           )
           np.savetxt("/app/x_train_pca.csv", self.x_train, delimiter = ',')
           np.savetxt("/app/x_test_pca.csv", self.x_test, delimiter = ',')
           # context.state.parameters_records['dados_transformados'] = ParametersRecord({'treino': self.x_train, 'teste': self.x_teste})
           return self.model.get_weights(), len(self.x_train), {'pca_exist': os.path.exists("/app/x_train_pca.csv")}
        else: 
           #self.x_train = np.array(self.client.state.parameters_records['dados_transformados']['dados_t_treino'])
           #self.x_test = np.array(self.client.state.parameters_records['dados_transformados']['dados_t_teste'])
           self.model.set_weights(parameters)
           self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
           )
           return self.model.get_weights(), len(self.x_train), {'pca_exist': os.path.exists("/app/x_train_pca.csv")}

    def evaluate(self, parameters, config):
           self.model.set_weights(parameters)
           loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
           # y_pred = self.model.predict(self.x_test, verbose = 0)
           # y_pred = np.argmax(y_pred, axis = 1)
           # conf_matrix = confusion_matrix(self.y_test, y_pred, labels = ['0', '1', '2', '3'])
           # conf_matrix = pd.DataFrame(conf_matrix, index = ['0', '1', '2', '3'], columns = ['0', '1', '2', '3'])
           # return loss, len(self.x_test), {"00": int(conf_matrix.iat[0,0]), "01": int(conf_matrix.iat[0,1]), "02": int(conf_matrix.iat[0,2]), "03": int(conf_matrix.iat[0,3]), "10": int(conf_matrix.iat[1,0]), "11": int(conf_matrix.iat[1,1]), "12": int(conf_matrix.iat[1,2]), "13": int(conf_matrix.iat[1,3]),"20": int(conf_matrix.iat[2,0]), "21": int(conf_matrix.iat[2,1]), "22": int(conf_matrix.iat[2,2]), "23": int(conf_matrix.iat[2,3]), "30": int(conf_matrix.iat[3,0]),"31": int(conf_matrix.iat[3,1]), "32": int(conf_matrix.iat[3,2]), "33": int(conf_matrix.iat[3,3])}
           return loss, len(self.x_test), {'accuracy': accuracy}
           # return 0.0, len(self.x_test), {'accuracy': 0.0}
def client_fn(context: Context):
    # Load model and data
    net = load_model()
    data = load_data()
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config["verbose"]

    # Return Client instance
    return FlowerClient(
        net, data, epochs, batch_size, verbose, context
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)

