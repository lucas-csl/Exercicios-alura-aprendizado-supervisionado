#!-*- coding: UTF-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression

# Lendo um arquivo em ".csv" com o pandas:
movies = pd.read_csv ("Datasets/regressao_linear_alura.csv")

x = movies ["Investimento (em milhoes)"]
y = movies ["Bilheteria (pessoas)"]

plt.scatter (x, y)
plt.show ()

# Determinando a amostra (Sample):
sample = movies.sample (n=200)
x_sample = sample ["Investimento (em milhoes)"]
y_sample = sample ["Bilheteria (pessoas)"]

plt.scatter (x_sample, y_sample)
plt.show ()

# Definindo treino e test:
filmes_investimento = movies ["Investimento (em milhoes)"]
filmes_bilheteria = movies ["Bilheteria (pessoas)"]

treino, teste, treino_bilheteria, teste_bilheteria = train_test_split (filmes_investimento, filmes_bilheteria)

# Precisamos transformas nossos dados em arrays:
treino = np.array(treino).reshape(len(treino), 1)
treino_bilheteria = np.array(treino_bilheteria).reshape(len(treino_bilheteria), 1)

teste = np.array(teste).reshape(len(teste), 1)
teste_bilheteria = np.array(teste_bilheteria).reshape(len(teste_bilheteria), 1)

# Definindo, treinando e testando nosso modelo:
modelo = LinearRegression ()
modelo.fit (treino, treino_bilheteria)

zootopia = [27.74456356]
bilheteria_zootopia = modelo.predict ([zootopia])
print modelo.score (teste, teste_bilheteria)
print "A bilheteria esperada para Zootopia utilizando este Algoritimo e de {0}".format(bilheteria_zootopia)