import numpy
import pandas as pd

endereco = ''
dados_cliente = pd.read_csv(endereco)
mcv = numpy.cov(dados_cliente.iloc[:,6:], rowvar = False)
numpy.save('mcv', mcv)
