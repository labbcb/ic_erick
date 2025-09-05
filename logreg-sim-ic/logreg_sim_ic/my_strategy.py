from typing import Union, Optional

# Reprodutibilidade
import random
random.seed(1)
import numpy as np
np.random.seed(1)
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.experimental.enable_op_determinism()
tf.random.set_seed(1)

from typing import Union, Optional
from flwr.common import FitRes, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
import joblib
from logreg_sim_ic.task import get_model, set_model_params

class CustomFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
        ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        results.sort(key=lambda x: x[1].metrics['partition-id'])
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        ndarrays = parameters_to_ndarrays(parameters_aggregated)
        model = get_model(penalty = "l2", local_epochs = 1)
        set_model_params(model, ndarrays)
        model.classes_ =  np.array([i for i in range(2)])
        joblib.dump(model, 'modelo_final_logreg_sim_ic')
        return parameters_aggregated, metrics_aggregated
