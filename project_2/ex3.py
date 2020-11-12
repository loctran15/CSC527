import numpy as np
import math
from random import random

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
        y1[i] = math.sqrt(random()) * math.sin(a) * w + (r * math.sin(a)) + d

        a = random() * math.pi + math.pi
        x2[i] = (r + w / 2) + math.sqrt(random()) * math.cos(a) * (w / 2) + (
            (-(r + w / 2)) if (random() < 0.5) else (r + w / 2)) * math.cos(a)
        y2[i] = -(math.sqrt(random()) * math.sin(a) * (-w) + (-r * math.sin(a))) - d
    return [x1, x2, y1, y2]


def activation_func(x, w, key):
    if(key == "ls"):
        activation = 0
        for i in range(3):
            activation += x[i] * w[i]
    elif(key == "perceptron"):
        activation = w[0]
        for i in range(len(x) - 1):
            activation += w[i + 1] * x[i]
    return 1.0 if activation >= 0.0 else -1.0


def get_normalize(dataset):
    normalized_data = np.asarray(dataset)
    sum_col = np.sum(normalized_data[:, :2], axis=0)
    mean_col = np.divide(sum_col, len(dataset))
    normalized_data[:, 0] = np.subtract(normalized_data[:, 0], mean_col[0])
    normalized_data[:, 1] = np.subtract(normalized_data[:, 1], mean_col[1])
    max_value = np.amax(abs(normalized_data[:, :2]))
    normalized_data[:, 0] = np.divide(normalized_data[:, 0], max_value)
    normalized_data[:, 1] = np.divide(normalized_data[:, 1], max_value)
    return normalized_data


def train_perceptron(dataset, epochs, learning_rate):
    dataset = np.copy(dataset[:, 1:])

    weights = [0 for _ in range(3)]
    mse_values = []

    for epoch in range(epochs):
        mse = 0.0
        for row in dataset:
            prediction = activation_func(row, weights,"perceptron")
            expected = row[-1]
            error = expected - prediction
            mse += error ** 2
            weights[0] += learning_rate * error
            for i in range(len(row) - 1):
                weights[i+1] += learning_rate * error * row[i]
        mse /= len(dataset)
        mse_values.append(mse)

        if mse == 0:
            break
    return mse_values, weights


def train_least_square(data, learning_rate):
    dataset = np.copy(data[:, :-1])
    label_set = np.copy(data[:, -1])
    I = np.identity(3)
    transposed_dataset = dataset.transpose()
    R = transposed_dataset.dot(dataset)
    R = np.reshape(R, (3, 3))
    weights = (np.linalg.inv(R + (learning_rate * I))).dot((transposed_dataset.dot(label_set)))

    MSE = 0
    for i in range(len(dataset)):
        error = abs(activation_func(dataset[i], weights, "ls") - label_set[i])
        MSE += error ** 2
    return MSE / len(dataset), weights


def train_LMS(dataset, epochs, eta):
    weights = np.zeros(2)
    mses = []
    for epoch in range(epochs):
        mse = 0.0
        np.random.shuffle(dataset)
        for row in dataset:
            row_without_label = row[:2]
            row_without_label = np.asarray(row_without_label)
            prediction = np.dot(weights, row_without_label)
            expected = row[-1]
            error = expected - prediction
            mse += error ** 2
            weights = weights + eta*error*row_without_label

        mse /= len(dataset)
        mses.append(mse)

        if mse == 0:
            break
    return mses, weights


if __name__ == "__main__":
    num_points = 1000
    d = 0
    lr = 0.1
    num_epochs = 50

    x1, x2, y1, y2 = moon(num_points, d, 10, 6)
    data = []
    data.extend([x1[i], y1[i], -1] for i in range(num_points))
    data.extend([x2[i], y2[i], 1] for i in range(num_points))
    data = np.asarray(data)

    ones = np.ones(num_points * 2).reshape(num_points * 2, 1)
    data = np.concatenate([ones, data], axis=1)
    normalize_data = get_normalize(np.copy(data[:,1:]))

    mse_ptr, weights_perceptron = train_perceptron(data, num_epochs, lr)
    mse_LS, weights_LS = train_least_square(data, lr)
    mse_LMS, weights_LMS = train_LMS(normalize_data, num_epochs, lr)

    print('Rosenblatt Perceptron algorithm -- ', end="")
    print('MSE: ', mse_ptr)
    print('Least Square algorithm -- ', end="")
    print('MSE: ', mse_LS)
    print('LMS algorithm -- ', end="")
    print('MSE: ', np.round_(mse_LMS, 3))
