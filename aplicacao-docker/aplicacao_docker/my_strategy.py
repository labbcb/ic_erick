from typing import Union, Optional
import numpy
from flwr.common import FitRes, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
import keras


from quickstart_docker_2.task import load_model
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
           local_means = [json.loads(r.metrics['local_mean'])  for _, r in results]
           local_sums_squares = [json.loads(r.metrics['local_sum_squares']) for _, r in results]
           global_mean = sum(local_means * examples) / sum (examples) 
           global_std = (sum(local_sums_squares) / sum(examples))**(0.5)
           return {}, {'global_mean': global_mean, 'global_std': global_std}

        else: 
           parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
           ndarrays = parameters_to_ndarrays(parameters_aggregated)
           model = load_model()
           model.set_weights(ndarrays)
           # model.save(filepath='modelo_docker_testando.keras')

           return parameters_aggregated, metrics_aggregated
