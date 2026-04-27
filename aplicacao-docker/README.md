# Aplicação TCGA em Docker

## Descrição dos arquivos

- Pasta `aplicacao-docker`: Contém os comandos em Python usados diretamente no treinamento de modelos de aprendizado de máquina:
    - `client_app.py`: Métodos analíticos utilizados localmente por cada cliente, incluindo pré-processamento, treinamento e avaliação de modelos;
    - `server_app.py`: Tarefas realizadas pelo agregador central, incluindo agregação de estatísticas sumárias (pré-processamento), modelos e métricas;
    - `task.py`: Arquivo auxiliar para definição de funções;
- `grid_search.sh`: Automatização da seleção de hiperparâmetros. Para cada combinação de hiperparâmetros, salva o resultado nos arquivos `results_rl.csv` (Regressão Linear) e `results_rn.csv` (Rede Neural).
- `pyproject.toml`: Arquivo de parâmetros do Flower, inclui dependências, arquivos componentes e definição de técnicas ou modelos e seus hiperparâmetros.
- `results_rl.csv` e `results_rn.csv`: Tabelas com resultados da seleção de hiperparâmetros dos modelos de Regressão Linear e de Rede Neural, respectivamente. Elas contém as colunas:
    - `tipo_dados`: Método de redução de dimensionalidade empregado;
    - `oversample`: Uso da técnica de oversample (`true` ou `false`); 
    - `penalty`: Regularização utilizada (`l1` ou `l2`);
    - `c`: Nível de regularização (0.01, 0.1, 1 ou 10);
    - `max_iter`: Número de rodadas;
    - `f1`: Métrica de Desempenho F1-Macro;
    - `loss`: Métrica Loss.
- `serverapp.Dockerfile`: arquivo utilizado para instalação das dependências.
- Pasta `objetos_intermediarios`: Contém arquivos referentes a medidas obtidas no pré-processamento e modelos federados finais:
    - `ap_global_rsv2.npy`: Valores singulares à direita globais, calculados por meio do método de [PCA Federada usando Agregação de Subespaços](https://doi.org/10.1093/bioadv/vbac026);
    - `g_mean.npy`: Médias de cada gene;
    - `g_std.npy`: Desvios padrão de cada gene;
    - `global_estimate.npy`: Estimativa global dos autovetores, calculados por meio do método de [Iteração de Subespaços Federada](https://doi.org/10.1093/bioadv/vbac026);
    - `rede_neural_tcga.keras`: Modelo de rede neural federado final
    - `reg_log_tcga`: Modelo de regressão logística federado final

## Instruções de Uso

1. Usando o arquivo [`start_flwr_suffix.sh`](start_flwr_suffix.sh), crie a rede de contêineres Docker para cada hospital participante:

```bash
~/ic_erick/start_flwr_suffix.sh A2 A8 AC AO AR B6 BH C8 D8 E2 EW
```

2. Altere os dados a serem lidos conforme a etapa que se deseja realizar (treinamento/validação) dentro da pasta [`aplicacao-docker`](aplicacao-docker/aplicacao_docker), no arquivo [`task.py`](aplicacao-docker/aplicacao-docker/task.py), na função `load_data` e no arquivo `client_app.py` na função `fit`. Dessa forma, os modelos são treinados no conjunto treino e testados no conjunto validação.

3. No arquivo [`pyproject.toml`](pyproject.toml), escolha o método de redução de dimensionalidade por meio da opção `algoritmo` e rode o programa:

```bash
flwr run . local-deployment --stream
```

4. Após cada contêiner possuir os dados transformados, defina os hiperparâmetros desejados e rode o arquivo `grid_search.sh`:

```bash
./grid_search.sh 
```

5. Novamente, altere os dados a serem lidos conforme a etapa que se deseja realizar (treinamento+validação/teste) dentro da pasta [`aplicacao-docker`](aplicacao-docker), no arquivo [`task.py`](aplicacao-docker/task.py), na função `load_data` e no arquivo `client_app.py` na função `fit`. Dessa forma, o modelo é treinado no conjunto treino+validação e testado no conjunto teste.

6. Novamente, no arquivo [`pyproject.toml`](pyproject.toml), escolha o método de redução de dimensionalidade por meio da opção `algoritmo` e rode o programa:

```bash
flwr run . local-deployment --stream
```

7. No arquivo [`pyproject.toml`](pyproject.toml), defina os hiperparâmetros de melhor desempenho em [`results_rl.csv`](results_rl.csv) ou [`results_rn.csv`](results_rn.csv) e rode o programa:

```bash
flwr run . local-deployment --stream
```

## Referências 

-[Quickstart with Docker](https://flower.ai/docs/framework/docker/tutorial-quickstart-docker.html)
-[Flower Network Communication](https://flower.ai/docs/framework/ref-flower-network-communication.html)
-[Federated horizontally partitioned principal component analysis for biomedical applications](https://doi.org/10.1093/bioadv/vbac026)
-[Ordem aleatória dos clientes](https://discuss.flower.ai/t/non-determinism-only-when-number-of-clients-is-increased/807)
