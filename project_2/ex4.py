from random import random
import matplotlib.pyplot as plt
import math
import numpy as np


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


def perceptron(dataset, epochs, learning_rate):
    weights = [0 for _ in range(3)]
    mse_values = []

    for epoch in range(epochs):
        mse = 0.0
        for row in dataset:
            prediction = activation_func(row, weights, "perceptron")
            expected = row[-1]
            error = expected - prediction
            mse += error ** 2
            weights[0] += learning_rate * error
            for i in range(len(row) - 1):
                weights[i+1] += learning_rate * error * row[i]
        mse = mse / len(dataset)
        mse_values.append(mse)

        if mse == 0:
            break
    return mse_values, weights


def LMS(dataset, epochs, eta):
    weights = np.random.rand(2)/2 - 0.25
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


def add_plot(mse, d):
    plt.plot(range(1, len(mse) + 1), mse, '--' , label=f"d={d}")
    plt.legend(loc="upper right")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title('Learning curve')

def get_normalize(dataset):
    normalized_data = np.asarray(dataset)
    sum_column = np.sum(normalized_data[:, :2], axis=0)
    mean_column = np.divide(sum_column, len(dataset))
    normalized_data[:, 0] = np.subtract(normalized_data[:, 0], mean_column[0])
    normalized_data[:, 1] = np.subtract(normalized_data[:, 1], mean_column[1])
    max_value = np.amax(abs(normalized_data[:, :2]))
    normalized_data[:, 0] = np.divide(normalized_data[:, 0], max_value)
    normalized_data[:, 1] = np.divide(normalized_data[:, 1], max_value)
    return normalized_data


if __name__ == "__main__":
    num_points = 1000
    num_epochs = 50
    width = 10
    radius = 6
    lr = 0.1
    ds = [0,1,-4]
    y_min = 10000
    y_max = 0

    for d in ds:
        x1, x2, y1, y2 = moon(num_points, d, width, radius)
        data = []
        data.extend([x1[i], y1[i], -1] for i in range(num_points))
        data.extend([x2[i], y2[i], 1] for i in range(num_points))
        data = np.asarray(data)
        normalize_data = get_normalize(np.copy(data))
        mse_LMS, weights = LMS(normalize_data, num_epochs, lr)
        add_plot(mse_LMS, d)
        mse_perceptron, weights_perceptron = perceptron(data, num_epochs, lr)
        print(f'-------------Distance d={d}---------------')
        print("Rosenblatt Perceptron algorithm -- ", end="")
        print('MSE: ', np.round_(mse_perceptron,4))
        print("LMS algorithm -- ", end="")
        print('MSE: ', np.round_(mse_LMS,4))
    plt.show()

