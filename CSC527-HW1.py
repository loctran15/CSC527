

#LOC TRAN
#CSC527

from copy import copy, deepcopy

def predict(row, weights):
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i + 1] * row[i]
    if activation >= 0:
        return 1
    else:
        return 0


def create_dataset(n, dataset, temp=[]):
    if (n == 0):
        dataset.append(temp)
        return
    for i in [0, 1]:
        temp.append(i)
        create_dataset(n - 1, dataset, list.copy(temp))
        temp.pop()

def add_ground_truth(dataset, keyword):
    if(keyword == "and"):
        for set in dataset:
            set.append(int(all(set)))
    elif(keyword == "or"):
        for set in dataset:
            set.append(int(any(set)))
    return dataset

if __name__ == "__main__":
    #part a
    print("PART A")
    n = 2
    dataset = []
    create_dataset(n, dataset)

    print("------------------------ n = 2, bias = -1 ------------------------")
    weights = [-1] + [1] * n
    print("dataset: ")
    dataset_or = add_ground_truth(deepcopy(dataset), "or")
    print(dataset_or)
    for row in dataset_or:
        prediction = predict(row, weights)
        print("Expected=%d, Predicted=%d" % (row[-1], prediction))

    print("------------------------ n = 2, bias = -2 ------------------------")
    weights = [-2] + [1] * n
    print("dataset: ")
    dataset_and = add_ground_truth(deepcopy(dataset), "and")
    print(dataset_and)
    for row in dataset_and:
        prediction = predict(row, weights)
        print("Expected=%d, Predicted=%d" % (row[-1], prediction))

    #part b

    print("PART B")
    n = 5
    dataset = []
    create_dataset(n, dataset)

    print("------------------------ n = 5, bias = -1 ------------------------")
    weights = [-1] + [1] * n
    print("dataset: ")
    dataset_or = add_ground_truth(deepcopy(dataset), "or")
    print(dataset_or)
    for row in dataset_or:
        prediction = predict(row, weights)
        print("Expected=%d, Predicted=%d" % (row[-1], prediction))

    print("------------------------ n = 5, bias = -5 ------------------------")
    weights = [-5] + [1] * n
    print("dataset: ")
    dataset_and = add_ground_truth(deepcopy(dataset), "and")
    print(dataset_and)
    for row in dataset_and:
        prediction = predict(row, weights)
        print("Expected=%d, Predicted=%d" % (row[-1], prediction))

    #part c
    print("")
    print("PART C")
    n = int(input("Enter n: "))
    dataset = []
    create_dataset(n, dataset)

    print("------------------------ n = {}, bias = -1 ------------------------".format(n))
    weights = [-1] + [1] * n
    print("dataset: ")
    dataset_or = add_ground_truth(deepcopy(dataset), "or")
    print(dataset_or)
    for row in dataset_or:
        prediction = predict(row, weights)
        print("Expected=%d, Predicted=%d" % (row[-1], prediction))

    print("------------------------ n = {}, bias = -{} ------------------------".format(n,n))
    weights = [-n] + [1] * n
    print("dataset: ")
    dataset_and = add_ground_truth(deepcopy(dataset), "and")
    print(dataset_and)
    for row in dataset_and:
        prediction = predict(row, weights)
        print("Expected=%d, Predicted=%d" % (row[-1], prediction))

    #pard d
    print("")
    print("PART D")
    print("the MP neuron will always be in the firing state if bias >= 0, the MP neuron will always be in the quiescent state when bias < -n. MP neuron produces AND logic if bias = -n and produces OR logic if bias = -1")
    print("Therefore, bias change based on the number of input signals")
