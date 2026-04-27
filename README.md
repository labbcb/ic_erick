# Programas Computacionais para Treinamento de Modelos de Classificação Federados

## Descrição

Esse repositório contém os programas computacionais desenvolvidos no projeto de iniciação científica **"Métodos de Aprendizado de máquina supervisionado e aprendizado federado para resolução de problemas de classificação na área da saúde"** realizado com apoio da **Fundação de Amparo à Pesquisa do Estado de São Paulo (FAPESP), Brasil** no processo nº 2024/20660-0. As opiniões, hipóteses e conclusões ou recomendações expressas neste material são de responsabilidade do(s) autor(es) e não necessariamente refletem a visão da FAPESP. 

Foram desenvolvidas duas aplicações com o emprego de modelos de **Aprendizado Federado (AF)** para resolução de problemas de classificação. A primeira realiza a **classificação de dados simulados**, com o objetivo de experimentar a ferramenta de implementação local do framework Flower (todo o sistema se encontra na mesma máquina) e controlar a natureza dos dados. A segunda se trata da **classificação de subtipos de câncer de mama com base em dados de sequenciamento de RNA (RNA-Seq)**. Para isso, foram coletados dados do The Cancer Genome Atlas (TCGA). Nesse caso, foi construída uma rede de contêineres Docker e os dados foram separados de acordo com sua origem a fim de experimentar o Flower numa estrutura federada. Além disso, para cada aplicação, foram desenvolvidos modelos de **Aprendizado de Máquina (AM)** centralizados análogos para comparação de desemepenho preditivo.

## Organização

### Classificação de Dados Simulados

- Pasta [`dados_simulacao`](dados_simulacao): contém os dados gerados por meio de simulação no arquivo [`IC_Simulacao.ipynb`](IC_Simulacao.ipynb);
- Pasta [`logreg-sim-ic`](logreg-sim-ic): contém os arquivos necessários para treinamento de modelos de regressão logística federado para os dados simulados;
- Pasta [`neural-sim-ic`](neural-sim-ic): contém os arquivos necessários para treinamento de modelos de rede neural federados para os dados simulados;
- Arquivo [`IC_Simulacao.ipynb`](IC_Simulacao.ipynb): contém geração de dados, treinamento de modelos de AM centralizados e avaliação dos modelos centralizados e federados.

### Classificação de Subtipos de Câncer de Mama

- Pasta [`aplicacao-docker`](aplicacao-docker): contém os arquivos necessários (exceto os dados) para treinamento de modelos federados para classificação de subtipos de câncer de mama;
- Arquivo [`TCGA_Centralizado.ipynb`](TCGA_Centralizado.ipynb): contém separação dos conjuntos de dados, pré-processamento e treinamento dos modelos de AM centralizados e avaliação dos modelos centralizados e federados.

