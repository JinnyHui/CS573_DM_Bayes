#!/usr/bin/env python 3
# Jingyi Hui, 11/04/2018
# CSCI573 Homework 3
# Implementation of Bayes Classifier
# Perform 3-fold cross validation


import numpy as np
from numpy import array
import pandas as pd
import csv
import JH_Bayes_Model
import JH_Bayes_Test
import operator
import sys


def train(train_set):
    """
    use training data to get the model
    :param train_set:
    :return: model parameters
    """
    line, column = train_set.shape
    class_dict = JH_Bayes_Model.class_generator(train_set, line)
    cardin = JH_Bayes_Model.cardinality(class_dict)
    prior = JH_Bayes_Model.prior_prob(cardin, line)
    mean = JH_Bayes_Model.sample_mean(class_dict)
    centered = JH_Bayes_Model.center_data(class_dict, mean)
    cov = JH_Bayes_Model.covariance_matrix(centered)
    model_3_fold = {}
    for label in keys:
        model_data = {}
        model_data['prior'] = prior[label]
        model_data['mean'] = mean[label]
        model_data['cov'] = cov[label]
        model_3_fold[label] = model_data
    return model_3_fold


def predict(test_set, model_3_fold):
    """
    Use Bayes classifier to predict data class
    :param test_set:
    :param model_3_fold:
    :return: a list of prediction
    """
    prediction_3_fold = []
    for instance in test_set:
        # print(instance)
        prediction_3_fold.append(JH_Bayes_Test.predict(instance, model_3_fold))
    # print('Predicted Label List:')
    # print(prediction_3_fold, '\n')
    return prediction_3_fold


def evaluate(prediction_3_fold, truth_label):
    """
    calculate and print all the evaluation
    :param prediction_3_fold: prediction list
    :param truth_label: truth label
    :return: a tuple of all the evaluations
    """
    y_actu = pd.Series(truth_label, name='Actual')
    y_pred = pd.Series(prediction_3_fold, name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred)
    print(df_confusion)
    key_list = set(truth_label)
    acc_dict = {}
    prec_dict = {}
    recall_dict = {}
    F_score_dict = {}
    for key in key_list:
        m = df_confusion.loc[key].sum()
        n = df_confusion[key].sum()
        nn = df_confusion.loc[key][key]
        prec_dict[key] = nn/m
        acc_dict[key] = prec_dict[key]
        recall_dict[key] = nn/n
        F_score_dict[key] = 2*nn/(n+m)
    return acc_dict, prec_dict, recall_dict, F_score_dict


#################################################################################
# load the data file
#################################################################################
file_name = sys.argv[1]
total_dataframe = pd.read_csv(file_name, header=None)
total_instance = len(total_dataframe)
first_end = int(total_instance/3)
second_end = int((total_instance/3) * 2)

fold_one = np.array(total_dataframe.iloc[:first_end])
# fold_one_label = total_dataframe.iloc[:first_end, -1]
fold_two = np.array(total_dataframe.iloc[first_end:second_end])
# fold_two_label = total_dataframe.iloc[first_end:second_end, -1]
fold_three = np.array(total_dataframe.iloc[second_end:])
# fold_three_label = total_dataframe.iloc[second_end:, -1]
dimension = fold_one.shape[1] - 1
keys = set(total_dataframe.iloc[:, -1])

#################################################################################
# perform 3-fold validation
#################################################################################
# fold one
model_one = train(np.concatenate((fold_one, fold_two)))
# print(model_one)
# write to csv file
with open('Model_3_Fold_1.csv', 'w') as file1:
    writer = csv.writer(file1)
    # write for each class
    for key, value in model_one.items():
        writer.writerow([key, value])
file1.close()

prediction_one = predict(fold_three, model_one)
# print(prediction_one)

# fold two
model_two = train(np.concatenate((fold_one, fold_three)))
# print(model_two)
# write to csv file
with open('Model_3_Fold_2.csv', 'w') as file2:
    writer = csv.writer(file2)
    # write for each class
    for key, value in model_two.items():
        writer.writerow([key, value])
file2.close()

prediction_two = predict(fold_two, model_two)
# print(prediction_two)

# fold three
model_three = train(np.concatenate((fold_two, fold_three)))
# print(model_three)
# write to csv file
with open('Model_3_Fold_3.csv', 'w') as file3:
    writer = csv.writer(file3)
    # write for each class
    for key, value in model_three.items():
        writer.writerow([key, value])
file3.close()

prediction_three = predict(fold_one, model_three)
# print(prediction_three)

#################################################################################
# print the evaluation
#################################################################################
# fold one
label_one = fold_three[:, -1].tolist()
print('##############################################################')
print('                 CONFUSION MATRIX FOLD_ONE                    ')
print('##############################################################')
fold_one_evaluation = evaluate(prediction_one, label_one)
accuracy_one, precision_one, recall_one, F_score_one = fold_one_evaluation
# print(fold_one_evaluation)

# fold two
label_two = fold_two[:, -1].tolist()
print('\n')
print('##############################################################')
print('                 CONFUSION MATRIX FOLD_TWO                    ')
print('##############################################################')
fold_two_evaluation = evaluate(prediction_two, label_two)
accuracy_two, precision_two, recall_two, F_score_two = fold_two_evaluation
# print(fold_two_evaluation)

# fold three
label_three = fold_one[:, -1].tolist()
print('\n')
print('##############################################################')
print('                CONFUSION MATRIX FOLD_THREE                   ')
print('##############################################################')
fold_three_evaluation = evaluate(prediction_three, label_three)
accuracy_three, precision_three, recall_three, F_score_three = fold_three_evaluation
# print(fold_three_evaluation)


#################################################################################
# average the evaluation over 3-folds
#################################################################################

for key in keys:
    print('\n')
    print('Class: ', key)
    avg_accuracy = "{0:.2f}".format((accuracy_one[key] + accuracy_two[key] + accuracy_three[key])/3)
    print('AVG Accuracy: ', str(avg_accuracy))
    avg_precision = "{0:.2f}".format((precision_one[key] + precision_two[key] + precision_three[key])/3)
    print('AVG Precision: ', str(avg_precision))
    avg_recall = "{0:.2f}".format((recall_one[key] + recall_two[key] + recall_three[key]) / 3)
    print('AVG Recall: ', str(avg_recall))
    avg_F_score = "{0:.2f}".format((F_score_one[key] + F_score_two[key] + F_score_three[key]) / 3)
    print('AVG F-Score: ', str(avg_F_score))
