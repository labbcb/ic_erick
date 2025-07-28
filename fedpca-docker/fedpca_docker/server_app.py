"""quickstart-docker-2: A Flower / PyTorch app."""

from typing import List, Tuple, Union, Optional
from flwr.common import Context, ndarrays_to_parameters, Metrics, FitRes, Parameters, Scalar, parameters_to_ndarrays, FitIns, EvaluateIns, EvaluateRes
from flwr.server import ServerApp, ServerAppComponents, ServerConfig, ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from fedpca_docker.task import load_model, load_data
# from quickstart_docker_2.my_strategy import CustomFedAvg
from sklearn.decomposition import TruncatedSVD
import numpy as np
import json

class CustomFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
        ) -> tuple[Optional[Parameters], dict[str, Scalar]]:

        if server_round == 1:
           examples = [r.num_examples for _, r in results]
           local_sums = [np.array(json.loads(r.metrics['local_sum'])) for _, r in results]
           local_sums_squares = [np.array(json.loads(r.metrics['local_sum_squares'])) for _, r in results]
           g_mean = sum(local_sums) / sum(examples)
           p1 = [(x**2)/sum(examples) for x in sum(local_sums)]
           g_var = np.array((sum(local_sums_squares) - p1) / (sum(examples) - 1))
           g_std = np.sqrt(g_var)
           self.global_mean = json.dumps(g_mean.tolist())
           self.global_std = json.dumps(g_std.tolist())
           self.k_components = min(examples)
           # parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
           # parameters_aggregated, metrics_aggregated = self.initial_parameters, {}
           parameters_aggregated, metrics_aggregated = ndarrays_to_parameters([g_mean, g_std]), {} 
           print(f"Rodada {server_round}: Calculadas médias e desvios padrões globais")

        elif server_round == 2:
           examples = [r.num_examples for _, r in results]
           local_sv = [np.array(json.loads(r.metrics['local_sv'])) for _, r in results]
           local_rsv = [np.array(json.loads(r.metrics['local_rsv'])) for _, r in results]
           t_local_rsv = [x.T for x in local_rsv]
           i = 0
           ap_global_cov = t_local_rsv[1] @ np.diag(local_sv[1]) @ local_rsv[1]
           """
           for trsv, sv, rsv in zip(t_local_rsv, local_sv, local_rsv):
               print(trsv.shape, sv.shape, rsv.shape)
               if i == 0:
                  ap_global_cov = (trsv @ np.diag(sv) @ rsv)              
               else:
                  ap_global_cov = ap_global_cov + (trsv @ np.diag(sv) @ rsv)
               i = i + 1
               print(f"Matriz de covariância do cliente {i}")
           """
           print('calculando SVD do global')
           svd = TruncatedSVD(n_components = min(examples) - 1, algorithm = 'arpack')
           svd.fit(ap_global_cov)
           self.ap_global_sv = json.dumps(svd.singular_values_.tolist())
           self.ap_global_rsv = json.dumps(svd.components_.tolist())
           parameters_aggregated, metrics_aggregated = self.initial_parameters, {}
           print(f"Rodada {server_round}: Calculadas sv e rsv globais")

        else: 
           parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
           ndarrays = parameters_to_ndarrays(parameters_aggregated)
           model = load_model()
           model.set_weights(ndarrays)
           # model.save(filepath='modelo_docker_testando.keras')
           print(f"Rodada {server_round}: Treinamento") 

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
           print(f"Rodada {server_round}: Solicitação de medidas locais (Padronização)")
        elif server_round == 2:
           config = {
            "current_round": server_round,
            "global_mean": self.global_mean,
            "global_std": self.global_std,
            "k_components": self.k_components,
           }
           print(f"Rodada {server_round}: Envio de médias e desvios padrão globais")
           print(f"Rodada {server_round}: Solicitação de medidas locais (PCA/SVD)")
        elif server_round == 3: 
           config = {
            "current_round": server_round,
            "ap_global_sv": self.ap_global_sv,
            "ap_global_rsv": self.ap_global_rsv,
           }
           print(f"Rodada {server_round}: Envio de sv e rsv globais")
        else:
           config = {
            "current_round": server_round,
           }
        print(config.keys())
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


# def fit_config(server_round: int):
    """Generate training configuration for each round."""
    # Create the configuration dictionary
    # if server_round == 1:
       # config = {
         # "current_round": server_round,
       # }
    # elif server_round == 2:
       # config = {
         # "current_round": server_round,
         # "global_mean": self.global_mean,
         # "global_std": self.global_std,
       # }
    # else:
       # config = {
         # "current_round": server_round,
         # "global_mean": self.global_mean,
         # "global_std": self.global_std,
       # }
    # return config

"""
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

        return loss_aggregated, metrics_aggregated

"""

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m['accuracy'] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    return {'Acurácia de Validação': sum(accuracies)/total_examples}

#def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # conf_matrix =[[m['00', '01', '02', '03', '10', '11', '12', '13', '20', '21', '22', '23', '30', '31', '32', '33'] for key in m] for _, m in metrics]
#    zz = [m['00'] for _,m in metrics]
#    return{'00' : zz}

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    # fraction_fit = context.run_config["fraction-fit"]
    fraction_fit = 1
    # Initialize model parameters
    parameters = ndarrays_to_parameters(load_model().get_weights())

    # Define strategy
    strategy = CustomFedAvg(
        fraction_fit = 1.0,
        fraction_evaluate = 1.0,
        min_available_clients = 11,
        initial_parameters = parameters,
        evaluate_metrics_aggregation_fn = weighted_average,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
