# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 19:51:05 2022

@author: Exaucé Maruba

J'analyse dans ce fichier Python les statistique d'un dataSet
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression



"""
le machine learning ML peut etre utilisé pour créer un chatbat,
reconnaitre les mauvais spams,la reconnaissance d'images,...

"""

#Statistiques sur les ages

ages = [15, 16, 18, 19, 22, 24, 29, 30, 34]
print(np.std(ages))
print(np.mean(ages))
print(np.percentile(25,ages))
print(np.percentile(75, ages))
print(np.var(ages))



""" 
DataFrame = table de données


"""

dataframe = pd.read_csv('titanic.csv')
#print(dataframe.head())
df = dataframe
#print(df.describe())
#print(df.head(4))
col = df['Fare']

"""
une serie Pandas est une colone unique d'un dataFrame, composé que d'une seule ligne

"""
#print(col)

small_df = df[['Fare','Survived']]
print(small_df.head(4))
print(small_df.describe())

df['Sex'] == 'male'
print(df.head(4))

"""
Création d'une serie Pandas nommée male avec valeur True si elle est male, False si faux 
dans la serie Sex'

df['male'] = df['Sex']=='male'
print(df.head())

serie = df['Fare'].values

print(serie, len(serie))


import time as t
for elt in range(0,len(serie)):
    
    print(f" {elt+1} / {len(serie)} est {serie[elt-1]}")
    t.sleep(0.1)
    
    
    
    arr = df[['Pclass', 'Fare', 'Age']].values
    masque = arr[:, 2] < 18

    enfants = arr[masque]


    Le masque permet de filitrer les données suivantes certaines conditions

    plt.scatter(df['Age'], df['Fare'] , c=df['Pclass'])
    plt.xlabel("Les Ages")
    plt.ylabel("Les Tarifs ")

    plt.plot([0,80], [85,5])
    print(enfants.shape)
    
    
"""


