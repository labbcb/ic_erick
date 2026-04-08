"""quickstart-docker-2: A Flower / PyTorch app."""
import os

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

import concurrent.futures
import io
import timeit
from logging import INFO, WARN
from flwr.common import (
    Code,
    DisconnectRes,
    ReconnectIns,
)
import importlib
from typing import List, Tuple, Union, Optional
from flwr.common import Context, ndarrays_to_parameters, Metrics, FitRes, Parameters, Scalar, parameters_to_ndarrays, FitIns, EvaluateIns, EvaluateRes
from flwr.common.logger import log
from flwr.common.typing import GetParametersIns
from flwr.server import ServerApp, ServerAppComponents, ServerConfig, ClientManager, Server
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from aplicacao_docker.task import load_data, load_model, get_model, get_model_params, set_initial_params, set_model_params, load_autoencoder_model, generate_random_gaussian, eigenvector_convergence_checker
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score, confusion_matrix
import json
import joblib
import scipy as sc
import scipy.linalg as la
from keras import Input, Model

class CustomFedAvg(FedAvg):
    def __init__(self, context, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context = context

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
        ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        results.sort(key=lambda x: x[1].metrics['client_id'])
        if self.context.run_config['algoritmo'] == 'AP-COV':
           if server_round == 1:
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
              np.save("/app/g_mean.npy", g_mean)
              np.save("/app/g_std.npy", g_std)
              self.global_mean = json.dumps(g_mean.tolist())
              self.global_std = json.dumps(g_std.tolist())
              # self.k_components = min(examples)
              self.k_components = 24
              parameters_aggregated, metrics_aggregated = None, {}
              print(f"RODADA {server_round}: CALCULADAS MÉDIAS E DESVIOS PADRÕES GLOBAIS")
           elif server_round == 2:
              examples = [r.num_examples for _, r in results]
              local_sv = [np.array(json.loads(r.metrics['local_sv'])) for _, r in results]
              local_rsv = [np.array(json.loads(r.metrics['local_rsv'])) for _, r in results]
              t_local_rsv = [x.T for x in local_rsv]
              i = 0
              for trsv, sv, rsv in zip(t_local_rsv, local_sv, local_rsv):
                  if i == 0:
                     ap_global_cov = (trsv @ np.diag(sv) @ rsv)
                  else:
                     ap_global_cov = ap_global_cov + (trsv @ np.diag(sv) @ rsv)
                  i = i + 1
                  print(f"CALCULADA MATRIZ DE COVARIÂNCIA APROXIMADA DO {i}º CLIENTE")
              #svd = TruncatedSVD(n_components = min(examples), algorithm = "randomized")
              svd = TruncatedSVD(n_components = 24, algorithm = "randomized")
              svd.fit(ap_global_cov)
              print('CALCULADA SVD GLOBAL APROXIMADA')
              np.save("/app/ap_global_sv2.npy", svd.singular_values_)
              np.save("/app/ap_global_rsv2.npy", svd.components_)
              self.ap_global_sv = json.dumps(svd.singular_values_.tolist())
              self.ap_global_rsv = json.dumps(svd.components_.tolist())
              parameters_aggregated, metrics_aggregated = None, {}
              print(f"Rodada {server_round}: CALCULADOS SV E RSV GLOBAIS")
           elif server_round == 3:
              parameters_aggregated, metrics_aggregated = None, {}
        elif self.context.run_config['algoritmo'] == 'SUB-IT':
           self.stop = (server_round > self.context.run_config["maxit"])
           if server_round == 1:
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
              np.save("/app/g_mean.npy", g_mean)
              np.save("/app/g_std.npy", g_std)
              self.global_mean = json.dumps(g_mean.tolist())
              self.global_std = json.dumps(g_std.tolist())
              #self.k_components = min(examples)
              converged = False
              self.increase_num_rounds_by = 1
              self.converged = converged
              X = np.array(generate_random_gaussian(m = 60660, k = self.context.run_config["n_eigenvectors"]))
              X, R = la.qr(X, mode = "economic")
              self.ge = json.dumps(X.tolist())
              parameters_aggregated, metrics_aggregated = None, {}
              print(f"RODADA {server_round}: CALCULADAS MÉDIAS E DESVIOS PADRÕES GLOBAIS")
                
           elif not self.converged and not self.stop:
              local_estimates = [np.array(json.loads(r.metrics['local_estimate'])) for _, r in results]
              ge_anterior = np.array(json.loads(self.ge))
              soma = sum(local_estimates)
              E = la.norm(soma, axis=0)
              ge, R = la.qr(soma, mode = "economic")
              converged, deltas, nr_converged = eigenvector_convergence_checker(ge, ge_anterior, tolerance = self.context.run_config["tolerance"])
              if converged:
                 print("CONVERGIU: SIM")
              else:
                 print("CONVERGIU: NÃO")
              print(f"CRITÉRIO DE COLINEARIDADE: {min(deltas)} (DEVE SER 1)")
              print(f"NÚMERO DE AUTOVETORES QUE CONVERGIRAM: {nr_converged}")
              self.stop = (server_round == self.context.run_config["maxit"])
              if self.stop:
                 print(f"ATINGIDO NÚMERO MÁXIMO DE ITERAÇÕES ({self.context.run_config['maxit']})")
              if converged or self.stop:
                 ord = np.argsort(E)
                 ge = np.flip(ge[:, ord], axis=1)
                 E = np.flip(np.sort(E))
                 np.save("/app/global_estimate.npy", ge)
                 self.increase_num_rounds_by = 0
              else:
                 self.increase_num_rounds_by = 1
              self.ge = json.dumps(ge.tolist())
              self.converged = converged
              parameters_aggregated, metrics_aggregated = None, {}
           else:
              parameters_aggregated, metrics_aggregated = None, {}
        elif self.context.run_config['algoritmo'] == 'Autoencoder':
           if server_round == 1:
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
              np.save("/app/g_mean.npy", g_mean)
              np.save("/app/g_std.npy", g_std)
              self.global_mean = json.dumps(g_mean.tolist())
              self.global_std = json.dumps(g_std.tolist())
              self.k_components = min(examples)
              parameters_aggregated, metrics_aggregated = None, {}
              print(f"RODADA {server_round}: CALCULADAS MÉDIAS E DESVIOS PADRÕES GLOBAIS")
           elif 2 <= server_round <= self.context.run_config['num-server-rounds'] - 1:
              parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
              ndarrays = parameters_to_ndarrays(parameters_aggregated)
              # Aggregate loss
              ac_loss_aggregated_train = weighted_loss_avg(
                  [
                      (evaluate_res.num_examples, evaluate_res.metrics['ac_loss_train'])
                      for _, evaluate_res in results
                  ]
              )
              ac_loss_aggregated_test = weighted_loss_avg(
                  [
                      (evaluate_res.num_examples, evaluate_res.metrics['ac_loss_test'])
                      for _, evaluate_res in results
                  ]
              )
              print(20*'-')
              print(f'LOSS PRÉ-AGREGAÇÃO DA RODADA {ac_loss_aggregated_train} (TREINO)')
              print(f'LOSS PRÉ-AGREGAÇÃO DA RODADA {ac_loss_aggregated_test} (VALIDAÇÃO)')
              print(20*'-')
              if server_round == self.context.run_config['num-server-rounds']-1:
                 autoencoder = load_autoencoder_model(input_size = self.context.run_config['input_size'], encoded_size = self.context.run_config['encoded_size'])
                 autoencoder.set_weights(ndarrays)
                 input_layer = Input(shape = (60660,))
                 bn_layer = autoencoder.get_layer(name = 'bn')(input_layer)
                 encoder = Model(input_layer, bn_layer)
                 encoder.save(filepath='/app/encoder_tcga.keras')
                 autoencoder.save(filepath="/app/autoencoder_tcga.keras")
           elif server_round == self.context.run_config['num-server-rounds']:
              parameters_aggregated, metrics_aggregated = None, {}
              
        elif self.context.run_config['algoritmo'] == 'Rede Neural':
           if server_round != self.context.run_config['num-server-rounds']:
              parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
              ndarrays = parameters_to_ndarrays(parameters_aggregated)
           else:
              parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
              ndarrays = parameters_to_ndarrays(parameters_aggregated)
              regularizer_function = getattr(importlib.import_module("tensorflow.keras.regularizers"), self.context.run_config['regularizer'])
              regularizer = regularizer_function(self.context.run_config['regularizer_lambda'])
              model = load_model(n_variaveis = self.context.run_config['n_variaveis'],
                                 hidden_layer_size = self.context.run_config['hidden_layer_size'],
                                 hidden_layer_num = self.context.run_config['hidden_layer_num'],
                                 regularizer = regularizer,
                                 n_classes = self.context.run_config['n_classes'])
              model.set_weights(ndarrays)
              model.save(filepath='rede_neural_tcga.keras')
           # Aggregate loss
           loss_aggregated_train = weighted_loss_avg(
               [
                   (evaluate_res.num_examples, evaluate_res.metrics['loss_train'])
                   for _, evaluate_res in results
               ]
           )
           loss_aggregated_test = weighted_loss_avg(
               [
                   (evaluate_res.num_examples, evaluate_res.metrics['loss_test'])
                   for _, evaluate_res in results
               ]
           )
           print(20*'-')
           print(f'LOSS PRÉ-AGREGAÇÃO DA RODADA {loss_aggregated_train} (TREINO)')
           print(f'LOSS PRÉ-AGREGAÇÃO DA RODADA {loss_aggregated_test} (VALIDAÇÃO)')
           conf_matrices_train = [np.array(json.loads(res.metrics['conf_matrix_train'])) for _, res in results]
           print(20*"-")
           print("MATRIZ DE CONFUSÃO GLOBAL PRÉ-AGREGAÇÃO DA RODADA (TREINO)")
           global_conf_m_train = sum(conf_matrices_train)
           print(global_conf_m_train)
           y_true_train, y_pred_train = [], []
           for i, linha in enumerate(global_conf_m_train):
              for j, qtd in enumerate(linha):
                 y_true_train.extend([i] * qtd)
                 y_pred_train.extend([j] * qtd)
           f1_macro_train = f1_score(np.array(y_true_train), np.array(y_pred_train), average = 'macro')
           print(f'F1-MACRO PRÉ-AGREGAÇÃO DA RODADA {f1_macro_train} (TREINO)')
           print(20*"-")
           print("MATRIZ DE CONFUSÃO GLOBAL PRÉ-AGREGAÇÃO DA RODADA (VALIDAÇÃO)")
           conf_matrices_test = [np.array(json.loads(res.metrics['conf_matrix_test'])) for _, res in results]
           global_conf_m_test = sum(conf_matrices_test)
           print(global_conf_m_test)
           y_true_test, y_pred_test = [], []
           for i, linha in enumerate(global_conf_m_test):
              for j, qtd in enumerate(linha):
                 y_true_test.extend([i] * qtd)
                 y_pred_test.extend([j] * qtd)
           f1_macro_test = f1_score(np.array(y_true_test), np.array(y_pred_test), average = 'macro')
           print(f'F1-MACRO PRÉ-AGREGAÇÃO DA RODADA {f1_macro_test} (VALIDAÇÃO)')
           print(20*"-")
           print(f"RODADA {server_round}: TREINAMENTO")
        elif self.context.run_config['algoritmo'] == 'Regressão Logística':
           if server_round != self.context.run_config['num-server-rounds']:
              parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
              ndarrays = parameters_to_ndarrays(parameters_aggregated)
           else:
              parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
              ndarrays = parameters_to_ndarrays(parameters_aggregated)
              model = get_model(penalty = self.context.run_config['penalty'],
                                C = self.context.run_config['C'],
                                solver = self.context.run_config['solver'],
                                max_iter = self.context.run_config['max_iter'])
              set_model_params(model, ndarrays)
              model.classes_ =  np.array([i for i in range(4)])
              joblib.dump(model, 'reg_log_tcga')
           # Aggregate loss
           loss_aggregated_train = weighted_loss_avg(
               [
                   (evaluate_res.num_examples, evaluate_res.metrics['loss_train'])
                   for _, evaluate_res in results
               ]
           )
           loss_aggregated_test = weighted_loss_avg(
               [
                   (evaluate_res.num_examples, evaluate_res.metrics['loss_test'])
                   for _, evaluate_res in results
               ]
           )
           print(20*'-')
           print(f'LOSS PRÉ-AGREGAÇÃO DA RODADA {loss_aggregated_train} (TREINO)')
           print(f'LOSS PRÉ-AGREGAÇÃO DA RODADA {loss_aggregated_test} (VALIDAÇÃO)')
           conf_matrices_train = [np.array(json.loads(res.metrics['conf_matrix_train'])) for _, res in results]
           print(20*"-")
           print("MATRIZ DE CONFUSÃO GLOBAL PRÉ-AGREGAÇÃO DA RODADA (TREINO)")
           global_conf_m_train = sum(conf_matrices_train)
           print(global_conf_m_train)
           y_true_train, y_pred_train = [], []
           for i, linha in enumerate(global_conf_m_train):
              for j, qtd in enumerate(linha):
                 y_true_train.extend([i] * qtd)
                 y_pred_train.extend([j] * qtd)
           f1_macro_train = f1_score(np.array(y_true_train), np.array(y_pred_train), average = 'macro')
           print(f'F1-MACRO PRÉ-AGREGAÇÃO DA RODADA {f1_macro_train} (TREINO)')
           print(20*"-")
           print("MATRIZ DE CONFUSÃO GLOBAL PRÉ-AGREGAÇÃO DA RODADA (VALIDAÇÃO)")
           conf_matrices_test = [np.array(json.loads(res.metrics['conf_matrix_test'])) for _, res in results]
           global_conf_m_test = sum(conf_matrices_test)
           print(global_conf_m_test)
           y_true_test, y_pred_test = [], []
           for i, linha in enumerate(global_conf_m_test):
              for j, qtd in enumerate(linha):
                 y_true_test.extend([i] * qtd)
                 y_pred_test.extend([j] * qtd)
           f1_macro_test = f1_score(np.array(y_true_test), np.array(y_pred_test), average = 'macro')
           print(f'F1-MACRO PRÉ-AGREGAÇÃO DA RODADA {f1_macro_test} (VALIDAÇÃO)')
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
           print(20*"-")
           print("PRINCIPAIS PARÂMETROS DE CONFIGURAÇÃO")
           print(f"Algoritmo: {self.context.run_config['algoritmo']}") 
           if self.context.run_config["algoritmo"] == "Regressão Logística" or self.context.run_config["algoritmo"] == "Rede Neural":
              print(f"Tipo de dados: {self.context.run_config['tipo_dados']} ({self.context.run_config['n_variaveis']} variáveis)")
           print(f"Oversample: {self.context.run_config['oversample']}")
           if self.context.run_config["algoritmo"] == "SUB-IT":
              print(f"Número de autovetores: {self.context.run_config['n_eigenvectors']}")
              print(f"Tolerância: {self.context.run_config['tolerance']}")
           print(20*"-")
        elif server_round == 2 and self.context.run_config['algoritmo'] == 'AP-COV':
           config = {
            "current_round": server_round,
            "global_mean": self.global_mean,
            "global_std": self.global_std,
            "k_components": self.k_components,
           }
           print(f"Rodada {server_round}: ENVIO DE MÉDIAS E DESVIOS PADRÃO GLOBAIS")
           print(f"Rodada {server_round}: SOLICITAÇÃO DE MEDIDAS LOCAIS (PCA/SVD)")
        elif server_round == 2 and self.context.run_config['algoritmo'] == "SUB-IT":
           config = {
            "current_round": server_round,
            "global_mean": self.global_mean,
            "global_std": self.global_std,
            "ge": self.ge,
           }
           print(f"Rodada {server_round}: ENVIO DE MÉDIAS E DESVIOS PADRÃO GLOBAIS")
        elif server_round == 2 and self.context.run_config['algoritmo'] == 'Autoencoder':
           config = {
            "current_round": server_round,
            "global_mean": self.global_mean,
            "global_std": self.global_std,
           }
           print(f"Rodada {server_round}: ENVIO DE MÉDIAS E DESVIOS PADRÃO GLOBAIS")
        elif server_round == 3 and self.context.run_config['algoritmo'] == 'AP-COV':
           config = {
            "current_round": server_round,
            "ap_global_sv": self.ap_global_sv,
            "ap_global_rsv": self.ap_global_rsv,
           }
           print(f"RODADA {server_round}: ENVIO DE SV E RSV GLOBAIS")
        elif self.context.run_config['algoritmo'] == "SUB-IT":
           config = {
            "current_round": server_round,
            "ge": self.ge,
            "converged": self.converged, 
            "stop": self.stop,
           }
           print(f"RODADA {server_round}: ENVIO DE AUTOVETORES GLOBAIS")
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
        if self.context.run_config["algoritmo"] == "AP-COV" or self.context.run_config["algoritmo"] == "SUB-IT":
           return []
        if self.fraction_evaluate == 0.0:
           return []

        # Parameters and config
        config = {
         "current_round": server_round,
        }

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

        results.sort(key=lambda x: x[1].num_examples)

        if self.context.run_config['algoritmo'] == 'AP-COV':
           return -1.0, {}
        elif self.context.run_config['algoritmo'] == 'Autoencoder':
           if server_round == 1:
              return -1.0, {}
           elif 2 <= server_round <= self.context.run_config['num-server-rounds'] - 1:
              # Aggregate loss
              ac_loss_aggregated_train = weighted_loss_avg(
                  [
                      (evaluate_res.num_examples, evaluate_res.metrics['ac_loss_train'])
                      for _, evaluate_res in results
                  ]
              )
              ac_loss_aggregated_test = weighted_loss_avg(
                  [
                      (evaluate_res.num_examples, evaluate_res.metrics['ac_loss_test'])
                      for _, evaluate_res in results
                  ]
              )
              print(20*'-')
              print(f'LOSS PÓS-AGREGAÇÃO DA RODADA = {ac_loss_aggregated_train} (TREINO)')
              print(f'LOSS PÓS-AGREGAÇÃO DA RODADA = {ac_loss_aggregated_test} (VALIDAÇÃO)')
              print(20*'-')
              return -1.0, {'loss de validação do autoencoder': ac_loss_aggregated_test}
           elif server_round == self.context.run_config['num-server-rounds']:
              return -1.0, {}
        elif self.context.run_config['algoritmo'] == 'Rede Neural' or self.context.run_config['algoritmo'] == 'Regressão Logística':
           # Aggregate loss
           loss_aggregated_train = weighted_loss_avg(
               [
                   (evaluate_res.num_examples, evaluate_res.metrics['loss_train'])
                   for _, evaluate_res in results
               ]
           )
           loss_aggregated_test = weighted_loss_avg(
               [
                   (evaluate_res.num_examples, evaluate_res.metrics['loss_test'])
                   for _, evaluate_res in results
               ]
           )
           print(20*'-')
           print(f'LOSS PÓS-AGREGAÇÃO DA RODADA = {loss_aggregated_train} (TREINO)')
           print(f'LOSS PÓS-AGREGAÇÃO DA RODADA = {loss_aggregated_test} (VALIDAÇÃO)')
           eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
           accuracies = [num_examples * m['accuracy_test'] for num_examples, m in eval_metrics]
           total_examples = sum(num_examples for num_examples, _ in eval_metrics)
           conf_matrices_train = [np.array(json.loads(res.metrics['conf_matrix_train'])) for _, res in results]
           print(20*"-")
           print("MATRIZ DE CONFUSÃO GLOBAL PÓS-AGREGAÇÃO DA RODADA (TREINO)")
           global_conf_m_train = sum(conf_matrices_train)
           print(global_conf_m_train)
           y_true_train, y_pred_train = [], []
           for i, linha in enumerate(global_conf_m_train):
              for j, qtd in enumerate(linha):
                 y_true_train.extend([i] * qtd)
                 y_pred_train.extend([j] * qtd)
           f1_macro_train = f1_score(np.array(y_true_train), np.array(y_pred_train), average = 'macro')
           print(f'F1-MACRO PÓS-AGREGAÇÃO DA RODADA = {f1_macro_train} (TREINO)')
           print(20*"-")
           print("MATRIZ DE CONFUSÃO GLOBAL PÓS-AGREGAÇÃO DA RODADA (VALIDAÇÃO)")
           conf_matrices_test = [np.array(json.loads(res.metrics['conf_matrix_test'])) for _, res in results]
           global_conf_m_test = sum(conf_matrices_test)
           print(global_conf_m_test)
           y_true_test, y_pred_test = [], []
           for i, linha in enumerate(global_conf_m_test):
              for j, qtd in enumerate(linha):
                 y_true_test.extend([i] * qtd)
                 y_pred_test.extend([j] * qtd)
           f1_macro_test = f1_score(np.array(y_true_test), np.array(y_pred_test), average = 'macro')
           print(f'F1-MACRO PÓS-AGREGAÇÃO DA RODADA = {f1_macro_test} (VALIDAÇÃO)')
           print(20*"-")
           # return loss_aggregated_test, {'Acurácia de Validação': sum(accuracies)/total_examples, 'F1-Macro': f1_macro}
           return loss_aggregated_test, {'F1-Macro': f1_macro_test}

#def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
#    accuracies = [num_examples * m['accuracy'] for num_examples, m in metrics]
#    total_examples = sum(num_examples for num_examples, _ in metrics)
#    return {'Acurácia de Validação': sum(accuracies)/total_examples}

# Custom Server 
class MyServer(Server):
     def fit(self, num_rounds: int, timeout: Optional[float]) -> tuple[History, float]:
         """Run federated averaging for a number of rounds."""
         history = History()

         # Initialize parameters
         log(INFO, "[INIT]")
         self.parameters = self._get_initial_parameters(server_round=0, timeout=timeout)
         log(INFO, "Starting evaluation of initial global parameters")
         res = self.strategy.evaluate(0, parameters=self.parameters)
         if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])
         else:
            log(INFO, "Evaluation returned no results (`None`)")

         # Run federated learning for num_rounds
         start_time = timeit.default_timer()

         if self.strategy.context.run_config["algoritmo"] == "SUB-IT":
            #for current_round in range(1, num_rounds + 1):
            #log(INFO, "")
            #log(INFO, "[ROUND %s]", current_round)
            num_rounds_left = num_rounds
            current_round = 0
            while num_rounds_left:
               current_round +=1
               res_fit = self.fit_round(
                   server_round=current_round,
                   timeout=timeout,
               )
               #print(f"Increasing rounds by: {self.strategy.increase_num_rounds_by}")
               num_rounds_left += self.strategy.increase_num_rounds_by

               if res_fit is not None:
                  parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                  if parameters_prime:
                     self.parameters = parameters_prime
                  history.add_metrics_distributed_fit(
                      server_round=current_round, metrics=fit_metrics
                  )

               # Evaluate model using strategy implementation
               res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
               if res_cen is not None:
                  loss_cen, metrics_cen = res_cen
                  log(
                      INFO,
                      "fit progress: (%s, %s, %s, %s)",
                      current_round,
                      loss_cen,
                      metrics_cen,
                      timeit.default_timer() - start_time,
                  )
                  history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                  history.add_metrics_centralized(
                      server_round=current_round, metrics=metrics_cen
                  )

               # Evaluate model on a sample of available clients
               res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
               if res_fed is not None:
                  loss_fed, evaluate_metrics_fed, _ = res_fed
                  if loss_fed is not None:
                     history.add_loss_distributed(
                      server_round=current_round, loss=loss_fed
                     )
                     history.add_metrics_distributed(
                       server_round=current_round, metrics=evaluate_metrics_fed
                     )
               num_rounds_left -=1
         else:
            for current_round in range(1, num_rounds + 1):
               log(INFO, "")
               log(INFO, "[ROUND %s]", current_round)
               # Train model and replace previous global model
               res_fit = self.fit_round(
                   server_round=current_round,
                   timeout=timeout,
               )
               if res_fit is not None:
                  parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                  if parameters_prime:
                     self.parameters = parameters_prime
                     history.add_metrics_distributed_fit(
                     server_round=current_round, metrics=fit_metrics
                  )

               # Evaluate model using strategy implementation
               res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
               if res_cen is not None:
                  loss_cen, metrics_cen = res_cen
                  log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                  )
                  history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                  history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                  )

               # Evaluate model on a sample of available clients
               res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
               if res_fed is not None:
                  loss_fed, evaluate_metrics_fed, _ = res_fed
                  if loss_fed is not None:
                     history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                     )
                     history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                     )

         # Bookkeeping
         end_time = timeit.default_timer()
         elapsed = end_time - start_time  
         return history, elapsed


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    # Initialize model parameters
    if context.run_config['algoritmo'] == 'Rede Neural':
       regularizer_function = getattr(importlib.import_module("tensorflow.keras.regularizers"), context.run_config['regularizer'])
       regularizer = regularizer_function(context.run_config['regularizer_lambda'])
       model = load_model(n_variaveis = context.run_config['n_variaveis'],
                          hidden_layer_size = context.run_config['hidden_layer_size'],
                          hidden_layer_num = context.run_config['hidden_layer_num'],
                          regularizer = regularizer,
                          n_classes = context.run_config['n_classes'])
       initial_parameters = ndarrays_to_parameters(model.get_weights())
    elif context.run_config['algoritmo'] == 'Regressão Logística':
       # Create LogisticRegression Model
       model = get_model(penalty = context.run_config['penalty'],
                         C = context.run_config['C'],
                         solver = context.run_config['solver'],
                         max_iter = context.run_config['max_iter'])
       # Setting initial parameters, akin to model.compile for keras models
       set_initial_params(model,
                          n_classes = context.run_config['n_classes'],
                          n_variaveis = context.run_config['n_variaveis'])
       initial_parameters = ndarrays_to_parameters(get_model_params(model))
    else:
       initial_parameters = None

    # Define strategy
    strategy = CustomFedAvg(
        context = context,
        fraction_fit = 1.0,
        fraction_evaluate = 1.0,
        initial_parameters = initial_parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    my_server = MyServer(client_manager = SimpleClientManager(), strategy = strategy)
    return ServerAppComponents(server = my_server, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)


