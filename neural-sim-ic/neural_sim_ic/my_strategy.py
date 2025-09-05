from typing import Union, Optional

import numpy
from flwr.common import FitRes, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
import keras


from neural_sim_ic.task import load_model


class CustomFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        ndarrays = parameters_to_ndarrays(parameters_aggregated)
        model = load_model()
        model.set_weights(ndarrays)
        model.save(filepath='scaled_mf2_200_neural_sim_ic_retreino.keras')

        return parameters_aggregated, metrics_aggregated