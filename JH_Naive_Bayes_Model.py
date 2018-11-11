#!/usr/bin/env python 3
# Jingyi Hui, 11/10/2018
# CSCI573 Homework 3
# Implementation of Naive Bayes Classifier
# Training Part

import numpy as np
import pandas as pd
import csv


def class_generator(dataset, line):
    """
    Generate class specific subsets from the dataset
    :param dataset: iris dataset with labels
    :param line: number of instance in the dataset
    :return: a dict of list which contains data instance belong to different classes
    """
    classdict = {}
    # line = dataset.shape[0]
    for i in range(line):
        class_label = dataset[i][-1]
        if class_label in classdict:
            classdict[class_label] = np.append(classdict[class_label], [dataset[i][:-1]], axis=0)
        else:
            classdict[class_label] = [dataset[i][:-1]]
    return classdict


def cardinality(dictionary):
    """
    calculate the cardinality of each class
    :param dictionary: a dict of list which contains data instance belong to different classes
    :return: a dictionary of cardinality of different classes
    """
    cardin_dict = {}
    for key in dictionary:
        cardin_dict[key] = dictionary[key].shape[0]
    return cardin_dict


def prior_prob(cardin_dict, total_instance):
    """
    calculate the prior probability for each class
    :param cardin_dict: cardinality for each class
    :param line: the total number of instance
    :return: a dictionary of prior probability for each class
    """
    prob_dict = {}
    for key in cardin_dict:
        prob_dict[key] = "{0:.2f}".format(cardin_dict[key]/total_instance)
    return prob_dict


def sample_mean(dictionary):
    """
    calculate the mean for each class
    :param dictionary:
    :return: a dictionary of mean for each class
    """
    mean_dict = {}
    for key in dictionary:
        mean_origin = np.mean(dictionary[key], axis=0)
        mean_dict[key] = [np.around(j, 2) for j in mean_origin]
    return mean_dict


def center_data(dictionary, mean_dict):
    """
    return centered dataset
    :param dictionary: instances in each class
    :param mean_dict: mean for each class
    :return: centered dataset for each class
    """
    centered_dict = {}
    for key in dictionary:
        centered_dict[key] = dictionary[key] - mean_dict[key]
    return centered_dict


def variance(centered_dict):
    """
    compute the covariance matrix for each centered class
    :param centered_dict: centered dataset for each class
    :return: variance for each centered class
    """
    variance_dict = {}
    for key in centered_dict:
        d = len(centered_dict[key][0])
        variance_list = []
        for index in range(d):
            variance_list.append(np.var(centered_dict[key][:, index]))
        variance_dict[key] = [np.around(j, 2) for j in variance_list]
    return variance_dict


if __name__ == '__main__':
    ###################################################################################
    # Start read in the data file
    ###################################################################################
    file_name = 'iris.txt.shuffled'
    dataframe = pd.read_csv(file_name, delimiter=',', header=None)
    keys = set(dataframe.iloc[:, -1])
    dataset = np.array(dataframe)
    line, column = dataset.shape
    class_dict = class_generator(dataset, line)  # class-specific subsets
    # print(classdict['Iris-versicolor'])
    # print(classdict['Iris-virginica'])
    # print(classdict['Iris-setosa'])
    cardin = cardinality(class_dict)  # cardinality
    # print(cardin)
    prior = prior_prob(cardin, line)  # prior prob
    # print(prior)
    mean = sample_mean(class_dict)  # mean
    # print(mean)
    centered = center_data(class_dict, mean)  # centered dataset
    variance = variance(centered)  # variance
    print('Model has been calculated, please check the output file: model_bayes.csv')

    #################################################################################
    # construct model dictionary
    #################################################################################
    model = {}
    for label in keys:
        model_data = {}
        model_data['prior'] = prior[label]
        model_data['mean'] = mean[label]
        # model_data['cov'] = cov[label]
        model_data['var'] = variance[label]
        model[label] = model_data
    print(model)

    #################################################################################
    # write the model to csv file
    #################################################################################
    with open('model_naive_bayes.csv', 'w') as file:
        writer = csv.writer(file)
        # write for each class
        for key, value in model.items():
            writer.writerow([key, value])
    file.close()
