from random import random
import matplotlib.pyplot as plt
import math
import numpy as np


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


def display_result(weights, mse, dataset):
    label_1_prediction_filter = dataset[:, 0] * weights[0] + dataset[:, 1] * weights[1] >= 0
    label_1_prediction_dataset = dataset[label_1_prediction_filter]
    label_2_prediction_filter = dataset[:, 0] * weights[0] + dataset[:, 1] * weights[1] < 0
    label_2_prediction_dataset = dataset[label_2_prediction_filter]

    n_epoch = 10 if len(mse) <= 10 else len(mse)

    plt.plot(range(1, len(mse)+1), mse, 'k-')
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title('Learning curve')
    plt.axis([1, n_epoch, 0, max(mse)])
    plt.ylim(np.min(mse), np.max(mse))
    plt.show()

    x = np.asarray([-20, 32])
    y = (-weights[0] * x)/weights[1]
    plt.plot(x, y, c="k")
    plt.xlim(-20, 32)
    plt.title('Testing result')
    plt.scatter(label_1_prediction_dataset[:, 0], label_1_prediction_dataset[:, 1], c="b", marker='x', s=20)
    plt.scatter(label_2_prediction_dataset[:, 0], label_2_prediction_dataset[:, 1], c="r", marker='x', s=20)
    plt.show()


def train(dataset, epochs, eta):
    weights = np.zeros(2)
    mses = []
    for epoch in range(epochs):
        mse = 0.0

        # shuffle the dataset for each epoch
        np.random.shuffle(dataset)
        for row in dataset:
            row_without_label = row[:2]
            row_without_label = np.asarray(row_without_label)
            prediction = np.dot(weights, row_without_label)
            expected = row[-1]
            error = expected - prediction
            mse += error ** 2
            weights = weights + eta*error*row_without_label

        mse = mse / len(dataset)
        mses.append(mse)

        if mse == 0:
            break
    return mses, weights


if __name__ == "__main__":
    total_points = 1000
    d = 1
    eta = 0.1
    n_epoch = 50
    x1, x2, y1, y2 = moon(total_points, d, 10, 6)
    data = []
    data.extend([x1[i], y1[i], -1] for i in range(total_points))
    data.extend([x2[i], y2[i], 1] for i in range(total_points))
    data = np.asarray(data)

    # normalize dataset
    normalized_data = np.asarray(np.copy(data))
    sum_column = np.sum(normalized_data[:, :2], axis=0)
    mean_column = np.divide(sum_column, len(normalized_data))
    normalized_data[:, 0] = np.subtract(normalized_data[:, 0], mean_column[0])
    normalized_data[:, 1] = np.subtract(normalized_data[:, 1], mean_column[1])
    max_value = np.amax(abs(normalized_data[:, :2]))
    normalized_data[:, 0] = np.divide(normalized_data[:, 0], max_value)
    normalized_data[:, 1] = np.divide(normalized_data[:, 1], max_value)

    # train the model and get mse and weights of the model.
    mse, weights = train(normalized_data, n_epoch, eta)

    display_result(weights, mse, data)
