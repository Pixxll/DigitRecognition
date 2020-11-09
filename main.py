import numpy as np
import data


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


inputs = np.array(data.data)


labels = np.array(data.labels)

np.random.seed(77)
weights = np.random.rand(2304, 10)
biases = np.random.rand(10)

outputs = []
for h in inputs:
    image = []
    for i in range(10):
        tmp = 0
        for j in range(len(h)):
            tmp += h[j] * weights[j][i]
        tmp += biases[i]
        image.append(max(tmp, 0))
    outputs.append(image)

print(outputs)




"""
[
[1, 3, 4 ... 2304],
[1, 3, 4],
[1, 3, 4],
[1, 3, 4],
[1, 3, 4],
[1, 3, 4],
[1, 3, 4]
... 100
]


"""
