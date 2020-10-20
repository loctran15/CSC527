# CSC 527
# Fall 2020
# Loc Tran

from random import random
import matplotlib.pyplot as plt
import math
import numpy as np

def predict(x, w):
    activation = 0
    for i in range(3):
        activation = activation + x[i] * w[i]
    if activation >= 0:
        return 1
    else:
        return -1

def get_weights(dataset,label_set, lamda):
    I = np.identity(3)
    transposed_dataset = dataset.transpose()
    R = transposed_dataset.dot(dataset)
    R = np.reshape(R, (3, 3))
    weights = (np.linalg.inv(R + (lamda * I))).dot((transposed_dataset.dot(label_set)))
    return weights

def display_result( weights, dataset):
    #classify the dataset into 2 subsets by label predictions
    label_1_prediction_filter = dataset[:,1]*weights[1] + dataset[:,2]*weights[2] >= -weights[0]
    label_1_prediction_dataset = dataset[label_1_prediction_filter]
    label_2_prediction_filter = dataset[:,1]*weights[1] + dataset[:,2]*weights[2] < -weights[0]
    label_2_prediction_dataset = dataset[label_2_prediction_filter]

    #plot decision boundary line
    min_x_value = -20
    max_x_value = 35
    x = np.asarray([min_x_value, max_x_value])
    y = -(weights[1]*x/weights[2]) - weights[0]/weights[2]
    plt.plot(x, y, c="g")

    #plot data points
    plt.scatter(label_1_prediction_dataset[:, 1], label_1_prediction_dataset[:, 2], c="r", s=10)
    plt.scatter(label_2_prediction_dataset[:, 1], label_2_prediction_dataset[:, 2], c="b", s=10)
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

def get_MSE(dataset, weights, label_set):
    MSE = 0
    for i in range(len(dataset)):
        error = abs(predict(dataset[i], weights) - label_set[i])
        MSE += error ** 2
    return MSE / len(dataset)



if __name__ == "__main__":
    #create a raw data set
    n_points = 1000
    distance = int(input("Enter the distance: "));
    raw_dataset = moon(n_points, distance, 10, 6)
    lamda = 0.1

    X1, X2, Y1, Y2 = raw_dataset[0], raw_dataset[1], raw_dataset[2], raw_dataset[3]
    ones = np.ones(n_points)
    minus_ones = np.ones(n_points)*-1
    label_set = np.concatenate((ones, minus_ones)).reshape((n_points*2, 1))
    x = X1 + X2
    x = np.asarray(x).reshape(n_points*2,1)
    y = Y1 + Y2
    y = np.asarray(y).reshape(n_points*2,1)
    x0 = np.ones((n_points*2,1))
    processed_dataset_without_labels = np.concatenate([x0, x, y],axis=1)


    #train the dataset and return Mean square root errors and weights
    weights = get_weights(processed_dataset_without_labels, label_set, lamda)
    MSE = get_MSE(processed_dataset_without_labels, weights, label_set)
    print("MSE: ", MSE)
    #plot the result
    display_result(weights, processed_dataset_without_labels)
