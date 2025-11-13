# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 11:00:16 2025

@author: alexa
"""

def train_model(train_X, train_y, test_X, model):
    """entraine un modele ML de classification et retourne des predictions sur
    le dataset de test"""
    print ("le train_model est prêt")
    model.fit(train_X,train_y) 
    prediction=model.predict(test_X)
    return prediction


