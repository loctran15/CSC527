# CSC/ECE/DA 427/527
# Fall 2020
# Loc Tran

from random import random
import matplotlib.pyplot as plt
import math
import numpy as np
from copy import deepcopy


def predict(row, weights):
    '''
    :param row: row of the dataset
    :param weights: weights of the perceptron
    :return: 0 or 1 value
    '''
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i+1] * row[i]
    return 1.0 if activation >= 0.0 else -1


def train_weights(dataset, learning_rate, n_epoch):
    '''
    :param dataset: the dataset we use to train perceptron
    :param learning_rate: The learning rate of the neural network
    :param n_epoch: number of epochs
    :return: mean square error list and weights of the neural network
    '''

    weights = [0.0 for i in range(len(dataset[0]))]
    MSE_list = []

    #for each epoch, calculate MSE and update the weights of the neural network
    for epoch in range(n_epoch):
        MSE = 0.0
        for row in dataset:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            MSE += error ** 2
            weights[0] = weights[0] + learning_rate * error
            for i in range(len(row) - 1):
                weights[i+1] = weights[i+1] + learning_rate * error * row[i]
        MSE = MSE / len(dataset)
        MSE_list.append(MSE)

        # if MSE is equal to 0, break the loop
        if MSE == 0.0:
            break
    return MSE_list, weights


def display_result(MSE_list, weights, dataset):
    '''
    :param MSE_list: list of mean square error
    :param weights:  weights of the neural network
    :param dataset:  The dataset used to plot
    :return: None
    '''
    #plot MSE in each epoch
    plt.plot(range(1, len(MSE_list) + 1),MSE_list)
    plt.ylabel("MSE")
    plt.xlabel("epoch")
    if (len(MSE_list) <= 10):
        n_epoch = 10
    else:
        n_epoch = len(MSE_list)
    plt.axis([1, n_epoch, 0, max(MSE_list)])
    plt.show()

    #classify the dataset into 2 subsets by label predictions
    label_1_prediction_filter = dataset[:,0]*weights[1] + dataset[:,1]*weights[2] >= -weights[0]
    label_1_prediction_dataset = dataset[label_1_prediction_filter]
    label_2_prediction_filter = dataset[:,0]*weights[1] + dataset[:,1]*weights[2] < -weights[0]
    label_2_prediction_dataset = dataset[label_2_prediction_filter]

    #plot decision boundary line
    min_x_value = -20
    max_x_value = 35
    x = np.asarray([min_x_value, max_x_value])
    y = -(weights[1]*x/weights[2]) - weights[0]/weights[2]
    plt.plot(x, y, c="g")

    #plot data points
    plt.scatter(label_1_prediction_dataset[:, 0], label_1_prediction_dataset[:, 1], c="r", s=10)
    plt.scatter(label_2_prediction_dataset[:, 0], label_2_prediction_dataset[:, 1], c="b", s=10)
    plt.xlim(min_x_value, max_x_value)
    plt.show()


def moon(num_points, distance, radius, width):
    points = num_points

    x1 = [0 for _ in range(points)]
    y1 = [0 for _ in range(points)]
    x2 = [0 for _ in range(points)]
    y2 = [0 for _ in range(points)]

    for i in range(points):
        d = distance
        r = radius
        w = width
        a = random() * math.pi
        x1[i] = math.sqrt(random()) * math.cos(a) * (w / 2) + (
                    (-(r + w / 2) if (random() < 0.5) else (r + w / 2)) * math.cos(a))
        y1[i] = math.sqrt(random()) * math.sin(a) * (w) + (r * math.sin(a)) + d

        a = random() * math.pi + math.pi
        x2[i] = (r + w / 2) + math.sqrt(random()) * math.cos(a) * (w / 2) + (
            (-(r + w / 2)) if (random() < 0.5) else (r + w / 2)) * math.cos(a)
        y2[i] = -(math.sqrt(random()) * math.sin(a) * (-w) + (-r * math.sin(a))) - d
    return ([x1, x2, y1, y2])




if __name__ == "__main__":
    #create a raw data set
    Learning_rate = 0.001
    raw_dataset = moon(1000, 0, 10, 6)
    raw_dataset = np.asarray(raw_dataset)

    # process the raw dataset
    tmp_dataset1 = np.resize(deepcopy(raw_dataset[0]), (1, len(raw_dataset[0])))
    tmp_dataset1 = np.append(tmp_dataset1, np.resize(deepcopy(raw_dataset[2]), (1, len(raw_dataset[0]))), axis=0)
    tmp_dataset1 = np.append(tmp_dataset1, np.ones((1, len(tmp_dataset1[0])))*-1, axis=0)
    tmp_dataset1 = tmp_dataset1.transpose()

    tmp_dataset2 = np.resize(deepcopy(raw_dataset[1]), (1, len(raw_dataset[1])))
    tmp_dataset2 = np.append(tmp_dataset2, np.resize(deepcopy(raw_dataset[3]), (1, len(raw_dataset[1]))), axis=0)
    tmp_dataset2 = np.append(tmp_dataset2, np.ones((1, len(tmp_dataset2[0]))), axis=0)
    tmp_dataset2 = tmp_dataset2.transpose()

    processed_dataset = np.append(tmp_dataset1, tmp_dataset2, axis=0)

    #shuffle the data set
    np.random.shuffle(processed_dataset)

    #train the dataset and return Mean square root errors and weights
    MSE_list, weights = train_weights(processed_dataset, Learning_rate, 100)

    #plot the result
    display_result(MSE_list, weights, processed_dataset)

