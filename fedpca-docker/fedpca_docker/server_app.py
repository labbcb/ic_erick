"""quickstart-docker-2: A Flower / PyTorch app."""

from typing import List, Tuple, Union, Optional
from flwr.common import Context, ndarrays_to_parameters, Metrics, FitRes, Parameters, Scalar, parameters_to_ndarrays, FitIns, EvaluateIns, EvaluateRes
from flwr.server import ServerApp, ServerAppComponents, ServerConfig, ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from fedpca_docker.task import load_model, load_data, get_model, get_model_params, set_initial_params
from sklearn.decomposition import TruncatedSVD
import numpy as np
import json
#import joblib

class CustomFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
        ) -> tuple[Optional[Parameters], dict[str, Scalar]]:

        if server_round == 1 and [r.metrics['pre_processamento'] for _, r in results] == ['Yes']*len(results):
           print('SERÃO REALIZADAS RODADAS DE PRÉ-PROCESSAMENTO')
           print('RECEBIDAS MÉDIAS E DESVIOS PADRÃO LOCAIS')
           examples = [r.num_examples for _, r in results]
           local_sums = [np.array(json.loads(r.metrics['local_sum'])) for _, r in results]
           local_sums_squares = [np.array(json.loads(r.metrics['local_sum_squares'])) for _, r in results]
           g_mean = sum(local_sums) / sum(examples)
           p1 = [(x**2)/sum(examples) for x in sum(local_sums)]
           p2 = sum(examples)*(g_mean**2)
           g_var = np.array((sum(local_sums_squares) - p1) / (sum(examples) - 1))
           g_var2 = np.array((sum(local_sums_squares) - p2) / (sum(examples) - 1))
           g_std = np.sqrt(g_var)
           g_std[g_std == 0] = 1
           np.save("g_mean.npy", g_mean)
           np.save("g_var.npy", g_var)
           np.save("g_var2.npy", g_var2)
           np.save("g_std.npy", g_std)
           self.global_mean = json.dumps(g_mean.tolist())
           self.global_std = json.dumps(g_std.tolist())
           self.k_components = min(examples)
           parameters_aggregated, metrics_aggregated = self.initial_parameters, {}
           self.tipo_dados = [r.metrics['tipo_dados'] for _, r in results][0]
           self.pre_processamento = [r.metrics['pre_processamento'] for _, r in results][0]
           print(f"RODADA {server_round}: CALCULADAS MÉDIAS E DESVIOS PADRÕES GLOBAIS")

        elif server_round == 2 and [r.metrics['pre_processamento'] for _, r in results] == ['Yes']*len(results):
           self.preprocessing = 'Yes'
           if [r.metrics['tipo_dados'] for _, r in results] == ['fedpca']*len(results):
              self.tipo_dados = 'fedpca'
              examples = [r.num_examples for _, r in results]
              local_sv = [np.array(json.loads(r.metrics['local_sv'])) for _, r in results]
              local_rsv = [np.array(json.loads(r.metrics['local_rsv'])) for _, r in results]
              t_local_rsv = [x.T for x in local_rsv]
              i = 0
              for trsv, sv, rsv in zip(t_local_rsv, local_sv, local_rsv):
                  print(trsv.shape, sv.shape, rsv.shape)
                  if i == 0:
                     ap_global_cov = (trsv @ np.diag(sv) @ rsv)
                  else:
                     ap_global_cov = ap_global_cov + (trsv @ np.diag(sv) @ rsv)
                  i = i + 1
                  print(f"CALCULADA MATRIZ DE COVARIÂNCIA APROXIMADA DO {i}º CLIENTE")
              print('CALCULANDO SVD GLOBAL APROXIMADA')
              svd = TruncatedSVD(n_components = min(examples) - 1, algorithm = 'arpack')
              svd.fit(ap_global_cov)
              np.save('g_components_ap',svd.components_)
              self.ap_global_sv = json.dumps(svd.singular_values_.tolist())
              self.ap_global_rsv = json.dumps(svd.components_.tolist())
              parameters_aggregated, metrics_aggregated = self.initial_parameters, {}
              print(f"Rodada {server_round}: CALCULADOS SV E RSV GLOBAIS")
           elif [r.metrics['tipo_dados'] for _, r in results] == ['autoencoder']*len(results):
              self.tipo_dados = 'autoencoder'
              parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
              ndarrays = parameters_to_ndarrays(parameters_aggregated)
              autoencoder = load_autoencoder_model()
              autoencoder.set_weights(ndarrays)
              autoencoder.save(filepath='autoencoder_docker_testando.keras')

        elif 3 <= server_round <= 6 and [r.metrics['tipo_dados'] for _, r in results] == ['autoencoder']*len(results) and [r.metrics['pre_processamento'] for _, r in results] == ['Yes']*len(results):
              self.tipo_dados, self.pre_processamento = 'autoencoder', 'Yes'
              parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
              ndarrays = parameters_to_ndarrays(parameters_aggregated)
              autoencoder = load_autoencoder_model()
              autoencoder.set_weights(ndarrays)
              autoencoder.save(filepath='autoencoder_docker_testando.keras')
        
        else: 
           if server_round == 1 and [r.metrics['pre_processamento'] for _, r in results] == ['No']*len(results):
              print('NÃO SERÃO REALIZADAS RODADAS DE PRÉ-PROCESSAMENTO')
           self.tipo_dados = [r.metrics['tipo_dados'] for _, r in results][0]
           self.pre_processamento = [r.metrics['pre_processamento'] for _, r in results][0]
           parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
           ndarrays = parameters_to_ndarrays(parameters_aggregated)
           if [r.metrics['metodo'] for _, r in results] == ['Rede Neural']*len(results): 
              print('MODELO DE REDE NEURAL')
              model = load_model()
              model.set_weights(ndarrays)
              model.save(filepath='modelo_docker_testando.keras')
           if [r.metrics['metodo'] for _, r in results] == ['Regressão Logística']*len(results):
              print("MODELO DE REGRESSÃO LOGÍSTICA")
              #model = get_model(penalty = penalty, local_epochs = epochs)
              #set_model_params(model, ndarrays)
              #model.classes_ =  np.array([i for i in range(4)])
              #joblib.dump(model, 'modelo_final_logreg_sim_ic')
           conf_matrices = [np.array(json.loads(res.metrics['conf_matrix'])) for _, res in results]
           print(20*"-")
           print("MATRIZ DE CONFUSÃO GLOBAL (TREINO)")
           print(sum(conf_matrices))
           print(20*"-")
           print(f"RODADA {server_round}: TREINAMENTO") 

        return parameters_aggregated, metrics_aggregated

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
        ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        # Create the configuration dictionary
        if server_round == 1:
           config = {
            "current_round": server_round,
           }
           print(f"RODADA {server_round}: SOLICITAÇÃO DE MEDIDAS LOCAIS (PADRONIZAÇÂO)")
        elif server_round == 2  and self.pre_processamento == 'Yes':
           if self.tipo_dados == 'fedpca':
              config = {
               "current_round": server_round,
               "global_mean": self.global_mean,
               "global_std": self.global_std,
               "k_components": self.k_components,
              }
              print(f"Rodada {server_round}: ENVIO DE MÉDIAS E DESVIOS PADRÃO GLOBAIS")
              print(f"Rodada {server_round}: SOLICITAÇÃO DE MEDIDAS LOCAIS (PCA/SVD)")
           elif self.tipo_dados == 'autoencoder':
              config = {
               "current_round": server_round,
               "global_mean": self.global_mean,
               "global_std": self.global_std,
              }
              print(f"Rodada {server_round}: ENVIO DE MÉDIAS E DESVIOS PADRÃO GLOBAIS")       
        elif server_round == 3 and self.tipo_dados == 'fedpca' and self.pre_processamento == 'Yes': 
           config = {
            "current_round": server_round,
            "ap_global_sv": self.ap_global_sv,
            "ap_global_rsv": self.ap_global_rsv,
           }
           print(f"RODADA {server_round}: ENVIO DE SV E RSV GLOBAIS")
        else:
           config = {
            "current_round": server_round,
           }
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]


    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
        ) -> list[tuple[ClientProxy, EvaluateIns]]:
        # Configure the next round of evaluation.
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
        ) -> tuple[Optional[float], dict[str, Scalar]]:
        # Aggregate evaluation losses using weighted average.
        
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
         
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")
        conf_matrices = [np.array(json.loads(res.metrics['conf_matrix'])) for _, res in results]
        print(20*"-")
        print("MATRIZ DE CONFUSÃO GLOBAL (VALIDAÇÃO)")
        print(sum(conf_matrices))
        print(20*"-")
        return loss_aggregated, metrics_aggregated


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m['accuracy'] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    return {'Acurácia de Validação': sum(accuracies)/total_examples}

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    # fraction_fit = context.run_config["fraction-fit"]
    fraction_fit = 1
    metodo = context.run_config['metodo']
    # Initialize model parameters
    if metodo == 'Rede Neural':
       initial_parameters = ndarrays_to_parameters(load_model().get_weights())
    elif metodo == 'Regressão Logística':
       # Create LogisticRegression Model
       penalty = context.run_config["penalty"]
       C = context.run_config['C']
       solver = context.run_config['solver']
       max_iter = context.run_config["max_iter"]
       model = get_model(penalty = penalty, C = C, solver = solver, max_iter = max_iter)
       # Setting initial parameters, akin to model.compile for keras models
       set_initial_params(model)
       initial_parameters = ndarrays_to_parameters(get_model_params(model))
 
    # Define strategy
    strategy = CustomFedAvg(
        fraction_fit = 1.0,
        fraction_evaluate = 1.0,
        min_available_clients = 11,
        initial_parameters = initial_parameters,
        evaluate_metrics_aggregation_fn = weighted_average,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
