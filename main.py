# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 10:58:40 2025

@author: alexa
"""

import pandas as pd 
from sklearn import svm  
from train_model import train_model
from preprocess_data import preprocess_data

iris = pd.read_csv(r"C:\Users\alexa\TP_Git\InPutData\Iris.csv") #load the dataset
test_size = 0.3 # the attribute test_size=0.3 to use for splitting the data 
				#into 70% for train and 30% for test

train, test =preprocess_data(iris, test_size)
# training data features
train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
# target of our training data
train_y=train.Species
# test data features
test_X= test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
#target value of test data
test_y =test.Species   


model = svm.SVC()
prediction = train_model(train_X, train_y, test_X, model)

from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

acc = accuracy_score(test_y, prediction)
print(f"Accuracy sur le test set : {acc:.2f}")

ConfusionMatrixDisplay.from_predictions(test_y, prediction)
import matplotlib.pyplot as plt
plt.show()
