## Leitura dos dados


import pandas as pd
a2 = pd.read_csv("/home/erick/dados_TCGA/fl_conj_teste_A2.csv")
dados_central = pd.read_csv("/home/erick/dados_TCGA/dados_TCGA_central/tpm_unstranded.csv")
# dados_central.head()
#dados_central.isna().sum()

### Quantidade de amostras por classe em cada cliente


contagens = dados_central.groupby(['tss', 'subtipo2']).size().to_frame(name = 'Count').reset_index().pivot(index = 'tss', columns='subtipo2', values='Count')
contagens = contagens.fillna(value=0)
contagens['Total'] = contagens['BRCA.Basal'] + contagens['BRCA.Her2'] + contagens['BRCA.LumA'] + contagens['BRCA.LumB']


### Clientes com 3 ou mais amostras em cada classe são incluídos no treinamento do AF


contagens['Função'] = contagens.gt(2).all(axis = 1).replace({True : 'Treino e Teste', False: 'Teste'})
contagens = contagens.sort_values(by= ['Função', 'Total'], ascending=False)
contagens['index'] = list(range(1,len(contagens)+1))
contagens = contagens.reset_index()
contagens



tt_tss = contagens.loc[contagens['Função'] == 'Treino e Teste', ['index','tss']] # Clientes treino e teste
t_tss = contagens.loc[contagens['Função'] == 'Teste', ['index','tss']] # Clientes apenas teste


### Divisão em Treino (60%), Validação (20%) e Teste (20%) feita de forma estratificada para garantir todas as classes em todas as divisões.


## Esse trecho foi executado no computador pessoal no Jupyter Notebook por conta da lentidão do Python nesse documento
from sklearn.model_selection import train_test_split
lista_conj_treino, lista_conj_valid, lista_conj_teste = [], [], []
for i in range(len(tt_tss['tss'])):
    dados_cliente = dados_central[dados_central['tss'] == tt_tss['tss'][i]]
    conj_treino_cliente, conj_teste_cliente = train_test_split(dados_cliente, test_size=0.2, random_state=i, stratify=dados_cliente['subtipo2'])
    conj_treino_cliente, conj_valid_cliente = train_test_split(conj_treino_cliente, test_size=0.25, random_state=1000+i, stratify=conj_treino_cliente['subtipo2'])
    # conj_treino_cliente.to_csv(f'fl_conj_treino_{tt_tss['tss'][i]}.csv', index = False)
    # conj_valid_cliente.to_csv(f'fl_conj_valid_{tt_tss['tss'][i]}.csv', index = False)
    # conj_teste_cliente.to_csv(f'fl_conj_teste_{tt_tss['tss'][i]}.csv', index = False)
    lista_conj_treino.append(conj_treino_cliente)
    lista_conj_valid.append(conj_valid_cliente)
    lista_conj_teste.append(conj_teste_cliente)
conj_treino = pd.concat(lista_conj_treino, ignore_index=True)
conj_valid = pd.concat(lista_conj_valid, ignore_index=True)
conj_teste = pd.concat(lista_conj_teste, ignore_index=True)


### Clientes que não participam do treino no AF são incluídos no conjunto de treino no AM. Dessa forma, os conjuntos de teste são iguais.


## Esse trecho foi executado no computador pessoal no Jupyter Notebook por conta da lentidão do Python nesse documento
for i in range(len(t_tss['tss'])):
    dados_cliente = dados_central[dados_central['tss'] == t_tss['tss'][i+11]]
    # dados_cliente.to_csv(f'fl_conj_total_{t_tss['tss'][i+11]}.csv', index = False)
    lista_conj_treino.append(dados_cliente)
conj_treino = pd.concat(lista_conj_treino, ignore_index=True)



# Leitura dos dados separados no computador pessoal

pd.read_csv("/home/erick/dados_TCGA/fl_conj_treino_A2.csv")



## Pré-processamento


from sklearn.preprocessing import LabelEncoder
x_treino, y_treino = conj_treino.iloc[:, 5:], conj_treino['subtipo2']
x_valid, y_valid = conj_valid.iloc[:, 5:], conj_valid['subtipo2']
x_teste, y_teste = conj_teste.iloc[:, 5:], conj_teste['subtipo2']
ee = LabelEncoder()
y_treino_enc = ee.fit_transform(y_treino)
y_valid_enc = ee.transform(y_valid)
y_teste_enc = ee.transform(y_teste)


### Padronização dos dados


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_treino_sc = scaler.fit_transform(x_treino)
x_valid_sc = scaler.transform(x_valid)
x_teste_sc = scaler.transform(x_teste)


### Autoencoder


import keras
from keras import layers, regularizers
input_layer = keras.Input(shape = (60660,))
encoded = layers.Dense(300, activation = 'relu', kernel_regularizer = regularizers.L1(1e-5))(input_layer)
decoded = layers.Dense(60660, activation='sigmoid')(encoded)
autoencoder = keras.Model(input_layer, decoded)
encoder = keras.Model(input_layer, encoded)
encoded_input = keras.Input(shape=(300,))
decoder_layer = autoencoder.layers[-1]
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
autoencoder.compile(optimizer=keras.optimizers.SGD(learning_rate = 1e-3), loss='mse')
autoencoder.fit(x_treino_sc, x_treino_sc,
                epochs=15,
                batch_size=256,
                validation_data=(x_valid_sc, x_valid_sc))



x_treino_ac = encoder.predict(x_treino_sc)
x_valid_ac = encoder.predict(x_valid_sc)
x_teste_ac = encoder.predict(x_teste_sc)


### Análise de Componentes Principais


from sklearn.decomposition import PCA
pca = PCA(random_state=1)
x_treino_pca = pca.fit_transform(x_treino_sc)
x_valid_pca = pca.transform(x_valid_sc)
x_teste_pca = pca.transform(x_teste_sc)
import matplotlib.pyplot as plt
plt.plot(pca.explained_variance_ratio_.cumsum(), marker = '.')
plt.xlabel('Componentes Principais')
plt.ylabel('Taxa de Variância Explicada')
plt.axhline(y = pca.explained_variance_ratio_.cumsum()[100], xmin = 0, xmax = 1000)
plt.axhline(y = pca.explained_variance_ratio_.cumsum()[200], xmin = 0, xmax = 1000)
plt.axhline(y = pca.explained_variance_ratio_.cumsum()[300], xmin = 0, xmax = 1000)
plt.show()



pca = PCA(n_components=300)
x_treino_pca = pca.fit_transform(x_treino_sc)
x_valid_pca = pca.transform(x_valid_sc)
x_teste_pca = pca.transform(x_teste_sc)


### Oversampling


from imblearn.over_sampling import ADASYN
adasyn_ac = ADASYN(random_state = 1, n_neighbors = 5)
x_treino_ac_resampled, y_treino_ac_resampled = adasyn_ac.fit_resample(x_treino_ac, y_treino)
adasyn_pca = ADASYN(random_state = 1, n_neighbors = 5)
x_treino_pca_resampled, y_treino_pca_resampled = adasyn_pca.fit_resample(x_treino_pca, y_treino)


## Grid Search


import numpy as np
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from scikeras.wrappers import KerasClassifier
from keras.callbacks import EarlyStopping


### Método de validação Holdout


ps = PredefinedSplit(np.hstack((np.ones(len(x_treino_ac_resampled))*(-1),np.ones(len(x_valid_ac))*0)))


### Função para gerar redes neurais para seleção de hiperparâmetros


def get_clf(hidden_layer_sizes, hidden_layer_num, regularizer):
    model = keras.Sequential()
    model.add(keras.Input(shape=(300,)))
    for i in range(hidden_layer_num):
        model.add(
            keras.layers.Dense(
                units=hidden_layer_sizes,
                activation='relu',
                kernel_regularizer = regularizer
        )
    )
    model.add(keras.layers.Dense(4, activation='softmax'))
    # model.compile(
    #     optimizer = keras.optimizers.SGD(learning_rate=learning_rate),
    #     loss='binary_crossentropy',
    #     metrics=['accuracy']
    return model


### Seleção de hiperparâmetros


model_params = {
    'log_reg': {
        'model': Pipeline([
                           ('log_reg', LogisticRegression())]),
        'params': {
            'log_reg__C': [0.01, 0.1, 1, 10],
            'log_reg__penalty': ['l1', 'l2'],
            'log_reg__solver': ['saga'],
            'log_reg__random_state': [1]
        }
    },
    'svm': {
        'model': Pipeline([
                           ('SVC', SVC())]),
        'params': {
            'SVC__C': [0.1, 1, 10, 100, 1000],
            'SVC__kernel': ['linear', 'poly', 'rbf'],
            'SVC__random_state': [1]
        }
    },
    'knn': {
        'model': Pipeline([
                           ('KNN', KNeighborsClassifier())]),
        'params': {
            'KNN__n_neighbors': [1, 3, 5, 15]
        }
    },
    'naive_bayes': {
        'model': Pipeline([
                           ('nb', GaussianNB())]),
        'params': {}
    },  
    'tree': {
        'model': Pipeline([
                           ('dt', DecisionTreeClassifier())]),
        'params': {
            'dt__max_depth': [1, 3, 10, 15, None],
            'dt__min_samples_split': [2, 6, 10],
            'dt__random_state': [1]
        }
    },
    'random_forest': {
        'model': Pipeline([
                           ('rf', RandomForestClassifier())]),
        'params': {
            'rf__max_depth': [1, 3, 10, 15, None],
            'rf__min_samples_split': [2, 6, 10], 
            'rf__random_state': [1]
        }
    },
    'nn': {
        'model': Pipeline([
                           ('keras', KerasClassifier(model=get_clf, random_state = 1, loss='sparse_categorical_crossentropy', optimizer='sgd', verbose=False, epochs = 15,
                                           callbacks=[EarlyStopping(patience=10, restore_best_weights=True,monitor='loss')]))]),
        'params':{
            'keras__optimizer__learning_rate': [1e-2, 1e-1],
            'keras__model__hidden_layer_sizes': [16, 32, 64, 128],
            'keras__model__hidden_layer_num': [1, 2, 3, 4, 5],
            'keras__model__regularizer': [regularizers.L1(1e-2), regularizers.L1(1e-3)]
        }
    }
}
scores = []
for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv = ps, return_train_score=False, scoring = 'f1_weighted')
    clf.fit(pd.concat([pd.DataFrame(x_treino_ac_resampled), pd.DataFrame(x_valid_ac)]), pd.concat([y_treino_ac_resampled, y_valid]))
    scores.append({
    'model': model_name,
    'best_score': clf.best_score_,
    'best_params': clf.best_params_
    })
validacao = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
validacao


## Avaliação dos modelos no conjunto de teste

### Ajuste dos modelos


rl = LogisticRegression(penalty='l1', C = 0.1, random_state=1, solver = 'saga')
rl.fit(pd.concat([pd.DataFrame(x_treino_pca), pd.DataFrame(x_valid_pca)]), pd.concat([y_treino, y_valid]))

svm = SVC(C = 100, kernel = 'rbf')
svm.fit(pd.concat([pd.DataFrame(x_treino_pca), pd.DataFrame(x_valid_pca)]), pd.concat([y_treino, y_valid]))

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(pd.concat([pd.DataFrame(x_treino_pca), pd.DataFrame(x_valid_pca)]), pd.concat([y_treino, y_valid]))

nb = GaussianNB()
nb.fit(pd.concat([pd.DataFrame(x_treino_sc), pd.DataFrame(x_valid_sc)]), pd.concat([y_treino, y_valid]))

nn = keras.Sequential()
nn.add(keras.Input(shape=(300,)))
nn.add(layers.Dense(16, activation='relu', kernel_regularizer = regularizers.L1(1e-2)))
nn.add(layers.Dense(16, activation='relu', kernel_regularizer = regularizers.L1(1e-2)))
nn.add(layers.Dense(16, activation='relu', kernel_regularizer = regularizers.L1(1e-2)))
nn.add(layers.Dense(4, activation="softmax"))
nn.compile(loss="sparse_categorical_crossentropy", optimizer = keras.optimizers.SGD(learning_rate=0.1), metrics = [keras.metrics.F1Score(average = 'weighted')])
nn.fit(pd.concat([pd.DataFrame(x_treino_pca), pd.DataFrame(x_valid_pca)]), pd.concat([pd.Series(y_treino_enc), pd.Series(y_valid_enc)]), epochs = 15, verbose = False, callbacks=[EarlyStopping(patience=10, restore_best_weights=True,monitor='loss')])

dt = DecisionTreeClassifier(max_depth = 10, min_samples_split = 10, random_state = 1)
dt.fit(pd.concat([pd.DataFrame(x_treino_sc), pd.DataFrame(x_valid_sc)]), pd.concat([y_treino, y_valid]))

rf = RandomForestClassifier(n_estimators = 100, max_depth = None, min_samples_split = 2, random_state = 1)
rf.fit(pd.concat([pd.DataFrame(x_treino_sc), pd.DataFrame(x_valid_sc)]), pd.concat([y_treino, y_valid]))


### Matrizes de Confusão e F1 ponderado


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.gridspec as gridspec
f1w = [sklearn.metrics.f1_score(y_teste, rl.predict(x_teste_pca), average = 'weighted'),
sklearn.metrics.f1_score(y_teste, svm.predict(x_teste_pca), average = 'weighted'),
sklearn.metrics.f1_score(y_teste, knn.predict(x_teste_pca), average = 'weighted'),
sklearn.metrics.f1_score(y_teste, nb.predict(x_teste_sc), average = 'weighted'),
sklearn.metrics.f1_score(y_teste, dt.predict(x_teste_sc), average = 'weighted'),
sklearn.metrics.f1_score(y_teste, rf.predict(x_teste_sc), average = 'weighted'),
sklearn.metrics.f1_score(y_teste_enc, np.argmax(nn.predict(x_teste_pca, verbose = 0), axis = 1), average = 'weighted')]
print(f1w)

cmrl = ConfusionMatrixDisplay(confusion_matrix(y_teste, rl.predict(x_teste_pca), normalize='true'), display_labels=['Basal', 'Her2', 'LumA', 'LumB'])
cmsvm = ConfusionMatrixDisplay(confusion_matrix(y_teste, svm.predict(x_teste_pca), normalize='true'), display_labels=['Basal', 'Her2', 'LumA', 'LumB'])
cmknn = ConfusionMatrixDisplay(confusion_matrix(y_teste, knn.predict(x_teste_pca), normalize='true'), display_labels=['Basal', 'Her2', 'LumA', 'LumB'])
cmnb = ConfusionMatrixDisplay(confusion_matrix(y_teste, nb.predict(x_teste_sc), normalize='true'), display_labels=['Basal', 'Her2', 'LumA', 'LumB'])      
cmdt = ConfusionMatrixDisplay(confusion_matrix(y_teste, dt.predict(x_teste_sc), normalize='true'), display_labels=['Basal', 'Her2', 'LumA', 'LumB'])
cmrf = ConfusionMatrixDisplay(confusion_matrix(y_teste, rf.predict(x_teste_sc), normalize='true'), display_labels=['Basal', 'Her2', 'LumA', 'LumB'])
cmnn = ConfusionMatrixDisplay(confusion_matrix(y_teste_enc, np.argmax(nn.predict(x_teste_pca, verbose = 0), axis = 1), normalize='true'), display_labels=['Basal', 'Her2', 'LumA', 'LumB'])
classificadores = [cmrl, cmsvm, cmknn, cmnb, cmdt, cmrf, cmnn]
nomes_c = ['Regressão Logística (com PCA)', 'Support Vector Machines (com PCA)', 'K Nearest Neighbors (com PCA)', 'Naive Bayes (sem PCA)', 'Árvore de Decisão (sem PCA)', 'Floresta Aleatória (sem PCA)', 'Redes Neurais (com PCA)']
for i in range(len(classificadores)):
    classificadores[i].plot()
    plt.title(nomes_c[i])
    plt.xlabel('Classe predita')
    plt.ylabel('Classe Verdadeira')
    plt.show()


