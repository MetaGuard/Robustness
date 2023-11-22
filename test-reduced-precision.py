# Import libraries
import numpy as np
from lib.model import run, split_data

# Load the input data
print("Loading Beat Saber data ...")
data = np.load('./data/boxrr.npy')
trainX, trainY, valX, valY, testX, testY = split_data(data)

# Test precision reductions
with open("./results/reduced-precision.csv", "w") as f:
    res = run("Round to 0.0001", trainX.round(4), trainY, testX.round(4), testY, valX.round(4), valY)
    f.write(res)

    res = run("Round to 0.001", trainX.round(3), trainY, testX.round(3), testY, valX.round(3), valY)
    f.write(res)

    res = run("Round to 0.01", trainX.round(2), trainY, testX.round(2), testY, valX.round(2), valY)
    f.write(res)

    res = run("Round to 0.1", trainX.round(1), trainY, testX.round(1), testY, valX.round(1), valY)
    f.write(res)

    res = run("Round to 1", trainX.round(0), trainY, testX.round(0), testY, valX.round(0), valY)
    f.write(res)
