#!-*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

# Lendo um arquivo em ".csv" com o pandas:
movies = pd.read_csv ("Datasets/avaliacoes_usuario.csv")

# Definindo treino e test:
caract = movies [movies.columns[1:16]]
gostos = movies [movies.columns[16:]]

treino, teste, treino_gostos, teste_gostos = train_test_split (caract, gostos, test_size = 0.1)

treino = np.array (treino).reshape(len(treino), 15)
teste = np.array (teste).reshape(len(teste), 15)

treino_gostos = treino_gostos.values.ravel()
teste_gostos = teste_gostos.values.ravel()

# Definindo, treinando e testando nosso modelo:
modelo = LogisticRegression ()
modelo.fit (treino, treino_gostos)

previsoes = modelo.predict (teste)
acuracia = accuracy_score (teste_gostos, previsoes)

print acuracia

modelo_NB = GaussianNB ()
modelo_NB.fit (treino, treino_gostos)

previsoes_NB = modelo_NB.predict (teste)
acuracia_NB = accuracy_score (teste_gostos, previsoes_NB)

print acuracia_NB
