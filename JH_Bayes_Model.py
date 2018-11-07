import numpy as np
import pandas as pd
import csv

# def class_generator(dataset):
#     """
#     Generate class specific subsets from the dataset
#     :param dataset: iris dataset with labels
#     :return: a dict of list which contains data instance belong to different classes
#     """
#     classdict = {}
#     class_label = set(dataset[:, -1])
#     list_length = len(class_label)
#     return classdict


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


def covariance_matrix(centered_dict):
    """
    compute the covariance matrix for each centered class
    :param centered_dict: centered dataset for each class
    :return: covariance matrix for each centered class
    """
    cov_dict = {}
    for key in centered_dict:
        cov_dict[key] = np.cov(centered_dict[key])
    return cov_dict


dataframe = pd.read_csv('iris.txt.shuffled', delimiter=',', header=None)
dataset = np.array(dataframe)
line, column = dataset.shape
class_dict = {}
for i in range(line):
    class_label = dataset[i][-1]
    if class_label in class_dict:
        class_dict[class_label] = np.append(class_dict[class_label], [dataset[i][:-1]], axis=0)
    else:
        class_dict[class_label] = [dataset[i][:-1]]
# print(classdict['Iris-versicolor'])
# print(classdict['Iris-virginica'])
# print(classdict['Iris-setosa'])
cardin = cardinality(class_dict)
print(cardin)
prior = prior_prob(cardin, line)
print(prior)
mean = sample_mean(class_dict)
print(mean)
centered = center_data(class_dict, mean)
cov = covariance_matrix(centered)
#################################################################################
# write the model to csv file
#################################################################################
with open('model_bayes.csv') as file:
    writer = csv.writer(file)
    # write prior probability of each class
    for key, value in prior.items():
        writer.writerow([key, value])
    # write mean of each class
    for key, value in mean.items():
        writer.writerow([key, value])
    # write covariance matrix for each class
    for key, value in cov.items():
        writer.writerow([key, value])
file.close()
