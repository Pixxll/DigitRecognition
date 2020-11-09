import numpy as np
import data


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Independent variables
input_set = np.array(data.data)

# Dependent variable
labels = np.array(data.labels)

# to convert labels to vector
labels = labels.reshape((100, 10))

np.random.seed(42)
weights = np.random.rand(2304, 10)
bias = np.random.rand(1)
lr = 0.05  # learning rate

for epoch in range(25000):
    inputs = input_set
    XW = np.dot(inputs, weights) + bias
    z = sigmoid(XW)
    error = z - labels
    print(error.sum())
    dcost = error
    dpred = sigmoid_derivative(z)
    z_del = dcost * dpred
    inputs = input_set.T
    weights = weights - lr * np.dot(inputs, z_del)

    for num in z_del:
        bias = bias - lr * num

single_pt = np.array([1, 1, 1])
result = sigmoid(np.dot(single_pt, weights) + bias)
print(result)
