#!/usr/bin/env python 3
# Jingyi Hui, 11/04/2018
# CSCI573 Homework 3
# Implementation of Bayes Classifier
# Testing Part

import numpy as np
from numpy import array
import pandas as pd
import csv
import operator


def predict(data, bayes):
    """
    for each data instance, calculate the posterior probabilities,
    and choose the maximum as prediction
    :param data: data instance
    :param bayes: a dict stores bayes model parameters
    :return: the prediction for the instance
    """
    result_dict = {}
    print('Instance:', str(data))
    for key in bayes:
        model_data = eval(bayes[key])
        determinant = np.linalg.det(model_data['cov'])
        centered_data = data[:dimension] - model_data['mean']
        exponent = - (np.dot(np.dot(centered_data.T, np.linalg.inv(model_data['cov'])), centered_data))/2
        denominator = ((np.sqrt(2*np.pi))**dimension) * np.sqrt(determinant)
        prior = float(model_data['prior'])
        posterior = ((1/denominator) * np.exp(exponent)) * prior
        # print(key, str(posterior))
        result_dict[key] = posterior
    prediction = max(result_dict.items(), key=operator.itemgetter(1))[0]
    print('Prediction:', str(prediction))
    print('True Label:', str(data[-1]))
    print('\n')
    return prediction


#################################################################################
# load the model from file
#################################################################################
with open('model_bayes.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)
    model = dict(reader)
    # print(model)


#################################################################################
# load the test file
#################################################################################
file_name = "test.csv"
dataframe = pd.read_csv(file_name, delimiter=',', header=None)
test_dataset = np.array(dataframe)
dimension = test_dataset.shape[1] - 1


#################################################################################
# Prediction
#################################################################################
predict_list = []
for instance in test_dataset:
    # print(instance)
    predict_list.append(predict(instance, model))
print('Predicted Label List:')
print(predict_list, '\n')


#################################################################################
# calculate the confusion matrix
#################################################################################
partition_k = len(model)
y_actu = pd.Series(test_dataset[:, -1], name='Actual')
y_pred = pd.Series(predict_list, name='Predicted')
print('##############################################################')
print('                       CONFUSION MATRIX                       ')
print('##############################################################')
df_confusion = pd.crosstab(y_actu, y_pred)
print(df_confusion)
