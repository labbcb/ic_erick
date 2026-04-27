# Simulação de Treinamento de Modelos de Rede Neural Federados

## Descrição dos arquivos:

- Pasta [`neural_sim_ic`](neural_sim_ic): Contém os comandos em Python usados diretamente no treinamento de modelos de aprendizado de máquina:
    - [`client_app.py`](neural_sim_ic/client_app.py): Métodos analíticos utilizados localmente por cada cliente, incluindo pré-processamento, treinamento e avaliação de modelos;
    - [`my_strategy.py`](neural_sim_ic/my_strategy.py): Personalização do processo de agregação;
    - [`server_app.py`](neural_sim_ic/server_app.py): Tarefas realizadas pelo agregador central, incluindo agregação de estatísticas sumárias (pré-processamento), modelos e métricas;
    - [`task.py`](neural_sim_ic/task.py`): Arquivo auxiliar para definição de funções;
- [`modelo_neural_sim_ic_final.keras`](modelo_neural_sim_ic_final.keras): Modelo final.
- [`pyproject.toml`](pyproject.toml): Arquivo de parâmetros do Flower, inclui dependências, arquivos componentes e definição de alguns parâmetros de treinamento.

## Instruções de Uso

1. Dentro da pasta `neural-sim-ic`, instale as dependências do projeto, contidas em [`pyproject.toml`](pyproject.toml):

```bash
pip install -e .
```

2. Dentro da pasta `neural-sim-ic`, rode a simulação local:

```bash
flwr run .
```
- **OBS.:** Baixe os conjuntos de dados contidos na pasta [`dados_simulacao`](../dados_simulacao) ou gere-os por meio do arquivo [`IC_Simulacao.ipynb`](../IC_Simulacao.ipynb).

## Referências

- [Quickstart Tensorflow](https://flower.ai/docs/framework/tutorial-quickstart-tensorflow.html)
- [Quickstart-tensorflow (Github)](https://github.com/flwrlabs/flower/tree/main/examples/quickstart-tensorflow)
