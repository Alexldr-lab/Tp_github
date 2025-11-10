# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 10:59:45 2025

@author: alexa
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data, testSize):
    """Nettoie, met en forme les données et prépare les ensembles de train et 
    de test"""
    
    # Suppression de la colonne 'Id'
    data = data.drop(columns=['Id'])
    
    # Normalisation des colonnes numériques
    numeric_cols = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
    scaler = MinMaxScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    
    # Split train/test
    train, test = train_test_split(data, test_size = testSize)
    return train, test
