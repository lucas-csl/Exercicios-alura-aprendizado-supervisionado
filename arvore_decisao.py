#!-*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

# Lendo um arquivo em ".csv" com o pandas:
movies = pd.read_csv ("Datasets/movies_multilinear_reg.csv")

# Definindo treino e test:
caract = movies [movies.columns[2:17]]
gostos = movies [movies.columns[17:]]

treino, teste, treino_gostos, teste_gostos = train_test_split (caract, gostos, test_size = 0.1)

treino = np.array (treino).reshape(len(treino), 15)
teste = np.array (teste).reshape(len(teste), 15)

treino_gostos = np.array (treino_gostos).reshape(len(treino_gostos), 1)
teste_gostos = np.array (teste_gostos).reshape(len(teste_gostos), 1)

# Definindo, treinando e testando nosso modelo:
modelo = tree.DecisionTreeRegressor ()
modelo.fit (treino, treino_gostos)

print modelo.score (treino, treino_gostos)
print modelo.score (teste, teste_gostos)

from sklearn.linear_model import LinearRegression

modelo_reg = LinearRegression ()
modelo_reg.fit (treino, treino_gostos)

print modelo_reg.score (treino, treino_gostos)
print modelo_reg.score (teste, teste_gostos)
