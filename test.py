import numpy as np

inputs = [
    [0, 1],
    [1, 0],
    [1, 1]
]

labels = [
    [1, 0],
    [0, 1],
    [0, 0]
]

labels = np.array(labels)

np.random.seed(77)
weights = np.random.rand(2, 2)
biases = np.random.rand(2)

for epoch in range(25000):
    output = []
    for set in inputs:
        array = []
        for out in range(2):
            tmp = 0
            for input in range(len(set)):
                tmp += set[input] * weights[input][out]
            tmp += biases[out]
            array.append(tmp)
        output.append(array)
    output = np.array(output)

    errors = []
    main_error = []
    for i in range(3):

        error = labels - output
        error1 = error[i][0] * weights[0][0] / (weights[0][0] + weights[0][1]) + error[i][1] * weights[0][1] / (weights[0][0] + weights[0][1])
        error2 = error[i][0] * weights[1][0] / (weights[1][0] + weights[1][1]) + error[i][1] * weights[1][1] / (weights[1][0] + weights[1][1])
        errors.append([error1, error2])

    for i in range(2):
        main_error.append((errors[0][i] + errors[1][i] + errors[2][i]) / 3)
    print(main_error)
    weights[0][0] += main_error[0] * 0.05
    weights[1][0] += main_error[0] * 0.05
    weights[0][1] += main_error[1] * 0.05
    weights[1][1] += main_error[1] * 0.05
print(output)

