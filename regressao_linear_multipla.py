#!-*- coding: UTF-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Lendo um arquivo em ".csv" com o pandas:
movies = pd.read_csv ("Datasets/movies_multilinear_reg.csv")

# Definindo treino e test:
filmes_independentes = movies [movies.columns[2:17]]
filmes_dependentes = movies [movies.columns[17:]]

treino, teste, treino_bilheteria, teste_bilheteria = train_test_split (filmes_independentes, filmes_dependentes)

# Definindo, treinando e testando nosso modelo:
modelo = LinearRegression ()
modelo.fit (treino, treino_bilheteria)

zootopia = [0,0,0,0,0,0,0,0,1,1,1,0,1,145.5170642,3.451632127]
planeta_dos_macacos = [0,1,0,0,0,0,0,0,0,0,0,0,0,150,5]
bilheteria_zootopia = modelo.predict ([zootopia])
bilheteria_pla_mac = modelo.predict ([planeta_dos_macacos])
print modelo.score (teste, teste_bilheteria)
print "A bilheteria esperada para Zootopia utilizando este Algoritimo e de {0}".format(bilheteria_zootopia)
print "A bilheteria esperada para Planeta dos Macacos utilizando este Algoritimo e de {0}".format(bilheteria_pla_mac)