import numpy as np
import math
import matplotlib.pyplot as plt


def get_dataset(a):
    # Generative model x(n) = a*x(n-1) + epsilon(n)
    mean, std_dvt = 0, math.sqrt(variance)
    epsilon = np.random.normal(mean, std_dvt, size=1000000)
    x = [0 for i in range(1000000)]
    x[0] = a * x0 + epsilon[0]
    for i in range(1, 1000000):
        if i == 1:
            x[1] = a * x[0] + epsilon[1]
        else:
            x[i] = a * x[i - 1] + epsilon[i]
    # get the last 5000 data points
    x = x[-5000:]
    return x

def display_result(J_theory, J_experiment):
    plt.legend(loc="upper right")
    plt.semilogy(J_theory, 'b--', label='Theory')
    plt.semilogy(J_experiment, 'k-', label='Experiment', linewidth=0.6)
    plt.title('eta = ' + str(eta))
    plt.xlabel("Number of iterations")
    plt.ylabel("MSE")
    plt.show()


if __name__ == '__main__':
    # Initialization
    n_data = 5000
    t = np.asarray(range(1, n_data+1,1))
    a = 0.99
    x0 = 0
    eta = 0.02
    sigu2 = 0.995
    variance = 0.01 #0.001 0.002
    errors = None
    n_epochs = 100

    for iteration in range(n_epochs):
        x_expected = get_dataset(a)
        x_expected = np.asarray(x_expected).reshape(n_data, 1)
        # Initialize weight matrix
        w = 0
        w0 = [0]
        w0 = np.asarray(w0)
        error = np.zeros((n_data, 1))

        # Predict
        x_predict = np.zeros((n_data, 1))
        x_predict[0, :] = x0 * w

        # Compute error: e(n) = d(n) - transpose(w) * x(n)
        error[0, :] = x_expected[0, :] - x_predict[0, :]

        # Compute weight: w(n+1) = w(n) + eta * x(n) * e(n)
        w = w + eta * error[0, :].T * x0

        # Loop through the rest of the dataset and compute the corresponding weights for each data point
        for n in range(1, n_data):
            w0 = np.append(w0, w)
            x_predict[n, :] = x_expected[n-1, :] * w
            error[n, :] = x_expected[n, :] - x_predict[n, :]
            w = w + eta * error[n, :].T * x_expected[n-1, :]

        if iteration == 0:
            errors = np.copy(error)
        else:
            errors = np.concatenate((errors, np.copy(error)), axis=0)

    # Reshape the error
    errors = errors.reshape((100, n_data))
    # LMS learning curve: formula 3.63 in the textbook
    J_theory = sigu2*(1-a**2)*(1+(eta/2)*sigu2) + sigu2*(a**2+(eta/2)*(a**2)*sigu2-0.5*eta*sigu2)*(1-eta*sigu2)**(2*t)
    # LMS mean square error: formula J = E(e(x)**2) in the textbook
    J_experiment = np.mean(np.square(errors), axis=0)

    display_result(J_theory, J_experiment)
