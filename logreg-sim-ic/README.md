# Simulação de Treinamento de Modelos de Regressão Logística Federados

## Descrição dos arquivos:

- Pasta [`logreg_sim_ic`](logreg_sim_ic): Contém os comandos em Python usados diretamente no treinamento de modelos de aprendizado de máquina:
    - [`client_app.py`](logreg_sim_ic/client_app.py): Métodos analíticos utilizados localmente por cada cliente, incluindo pré-processamento, treinamento e avaliação de modelos;
    - [`my_strategy.py`](logreg_sim_ic/my_strategy.py): Personalização do processo de agregação;
    - [`server_app.py`](logreg_sim_ic/server_app.py): Tarefas realizadas pelo agregador central, incluindo agregação de estatísticas sumárias (pré-processamento), modelos e métricas;
    - [`task.py`](logreg_sim_ic/task.py`): Arquivo auxiliar para definição de funções;
- [`modelo_final_logreg_sim_ic`](modelo_final_logreg_sim_ic): Modelo final.
- [`pyproject.toml`](pyproject.toml): Arquivo de parâmetros do Flower, inclui dependências, arquivos componentes e definição de alguns parâmetros de treinamento.

## Instruções de Uso

1. Dentro da pasta `logreg-sim-ic`, instale as dependências do projeto, contidas em [`pyproject.toml`](pyproject.toml):

```bash
pip install -e .
```

2. Dentro da pasta `logreg-sim-ic`, rode a simulação local:

```bash
flwr run .
```
- **OBS.:** Baixe os conjuntos de dados contidos na pasta [`dados_simulacao`](../dados_simulacao) ou gere-os por meio do arquivo [`IC_Simulacao.ipynb`](../IC_Simulacao.ipynb).

## Referências

- [Quickstart scikit-learn](https://flower.ai/docs/framework/tutorial-quickstart-scikitlearn.html)
- [Quickstart-sklearn (Github)](https://github.com/flwrlabs/flower/tree/main/examples/quickstart-sklearn)