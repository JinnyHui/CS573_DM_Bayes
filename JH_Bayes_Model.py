import numpy as np
import pandas as pd

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


def prior_prob(cardin_dict, line):
    """
    calculate the prior probability for each class
    :param cardin_dict: cardinality for each class
    :param line: the total number of instance
    :return: a dictionary of prior probability for each class
    """
    prob_dict = {}
    for key in cardin_dict:
        prob_dict[key] = "{0:.2f}".format(cardin_dict[key]/line)
    return prob_dict

def mean():


    """
    calculate the mean for each class
    :return:
    """
    return mean_dict
dataframe = pd.read_csv('iris.txt.shuffled', delimiter=',', header=None)
dataset = np.array(dataframe)
line, column = dataset.shape
classdict = {}
for i in range(line):
    class_label = dataset[i][-1]
    if class_label in classdict:
        classdict[class_label] = np.append(classdict[class_label], [dataset[i][:-1]], axis=0)
    else:
        classdict[class_label] = [dataset[i][:-1]]
print(classdict['Iris-versicolor'])
print(classdict['Iris-virginica'])
print(classdict['Iris-setosa'])
cardin = cardinality(classdict)
print(cardin)
prior = prior_prob(cardin, line)
print(prior)
