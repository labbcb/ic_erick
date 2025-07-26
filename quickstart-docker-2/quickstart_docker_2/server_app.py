"""quickstart-docker-2: A Flower / PyTorch app."""

from typing import List, Tuple
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from quickstart_docker_2.task import load_model, load_data

from quickstart_docker_2.my_strategy import CustomFedAvg
import numpy as np

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
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=6,
        initial_parameters=parameters,
	evaluate_metrics_aggregation_fn=weighted_average,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
