from typing import Union, Optional

import numpy as np
from flwr.common import FitRes, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
import joblib
from quickstart_docker_logreg.task import get_model_params, get_model, set_model_params

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
        # numpy.concatenate(ndarrays[0], ndarrays[1])
        model = get_model(penalty = "l1", local_epochs = 5)
        set_model_params(model, ndarrays)
        model.classes_ =  np.array([i for i in range(4)])
        joblib.dump(model, 'modelo_final_logreg_sim_ic')
        return parameters_aggregated
