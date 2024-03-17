#Imports
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.metrics import accuracy_score

#Importando dados
dados = pd.read_csv("dataset.csv")

#Análise Exploratória
print(dados.dtypes)
print(dados.describe)

dados.hist(bins = 10)
plt.show()
#As variáveis Alamine_Aminotransferase e Aspartate_Aminotransferase parecem possuir outliers
#pois quando comparamos o histograma com suas médias e valores máximos, os valores são muito distuantes

#Transformando a variável target (dataset) em binária
def ajusta_target(x):
    if x == 2:
        return 0
    return 1

dados['Dataset'] = dados['Dataset'].map(ajusta_target)
dados.rename({'Dataset': 'Target'}, axis = 'columns', inplace = True)
print(dados.sample(5))

#Transformando a variável categórica em numérica para posteriormente aplicarmos os algoritmos
def encoding_func(x):
    if x == 'Male':
        return 0
    return 1

dados['Gender'] = dados['Gender'].map(encoding_func) 
print(dados.sample(5))

#Verificando valores null e duplicados
print(dados[dados.isnull().values])
print(dados[dados.duplicated(keep = False)])

#Removendo duplicatas
dados = dados.drop_duplicates()

#Tratando os outliers
print(dados.describe())

sns.boxplot(dados.Alamine_Aminotransferase)
plt.show()

#Contagem de frequências da variável acima, para identificar se realmente é um outlier
print(dados.Alamine_Aminotransferase.sort_values(ascending = False).head()) #Não iremos remover, pois senão seriam 117 clientes removidos

#Verificando outra variável
sns.boxplot(dados.Aspartate_Aminotransferase)
plt.show()
#Contagem de frequências
print(dados.Alamine_Aminotransferase.sort_values(ascending = False).head()) 

dados = dados[dados.Alamine_Aminotransferase <= 2500] #Removendo do dataframe linhas dessa variável que seja maior que 2500
print(dados.shape)

#Tratamento de valores ausentes
print(dados[dados.isnull().values])
dados = dados.dropna(how = 'any')
