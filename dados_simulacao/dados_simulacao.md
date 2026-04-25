# Dados de Simulação

Descrição dos arquivos:
- `IC_S_fl_conj_treino_total.csv`: conjunto de treino
- `IC_S_fl_conj_valid_total.csv`: conjunto de validação
- `IC_S_fl_conj_treino_final.csv`: conjunto de treino + validação

Cada arquivo possui colunas:
- `x1`: Variável Preditora
- `x2`: Variável Preditora
- `Resposta`: Variável Resposta (0 ou 1)
- `Cliente`: Utilizada para federação dos dados (1, 2 ou 3)

Os dados foram simulados de acordo com a tabela e as equações abaixo:

<center>

| *Cliente* | $$\boldsymbol{x_{1}}$$ | $$\boldsymbol{x_{2}}$$ | $$\boldsymbol{\alpha}$$ | $$\boldsymbol{\beta_{1}}$$ | $$\boldsymbol{\beta_{2}}$$ | $$\boldsymbol{\epsilon}$$ | $$\boldsymbol{n}$$ |
| --- | --- | --- | --- | --- | --- | --- | --- | 
| 1 | $$U(-10, 10)$$ | $$U(0, 20)$$ | $$1.5$$ | $$2.5$$ | $$-2.0$$ | $$N(0, 16)$$ | $$30$$ |
| 2 | $$U(0, 10)$$ | $$U(0, 10)$$ | $$1.5$$ | $$2.6$$ | $$-1.9$$ | $$N(0, 16)$$ | $$40$$ |
| 3 | $$U(-10, 10)$$ | $$U(-5, 15)$$ | $$1.3$$ | $$2.5$$ | $$-2.1$$ | $$N(0, 16)$$ | $$50$$ |



$$ \eta _{ij}= \alpha_ix_{1ij} + \beta_{1i}x_{2ij} + \beta_{2i} + \varepsilon_{ij}; i = 1,2,3; j = 1, ..., n_i $$

$$ p _{ij} = \frac{1}{1+e^{-\eta _{ij}}} \text{ (Função Logística)} $$

$$ y_{ij} \sim Bernoulli(p_{ij}) $$

</center>
