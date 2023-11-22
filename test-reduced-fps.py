# Import libraries
import numpy as np
from lib.model import run, split_data

# Load the input data
print("Loading Beat Saber data ...")
data = np.load('./data/boxrr.npy')
trainX, trainY, valX, valY, testX, testY = split_data(data)

# Test FPS reductions
with open("./results/reduced-fps.csv", "w") as f:
    frames = np.arange(0, 900, 1)
    res = run("30 FPS", trainX[:,frames,:], trainY, testX[:,frames,:], testY, valX[:,frames,:], valY)
    f.write(res)

    frames = np.arange(0, 900, 2)
    res = run("15 FPS", trainX[:,frames,:], trainY, testX[:,frames,:], testY, valX[:,frames,:], valY)
    f.write(res)

    frames = np.arange(0, 900, 3)
    res = run("10 FPS", trainX[:,frames,:], trainY, testX[:,frames,:], testY, valX[:,frames,:], valY)
    f.write(res)

    frames = np.arange(0, 900, 6)
    res = run("5 FPS", trainX[:,frames,:], trainY, testX[:,frames,:], testY, valX[:,frames,:], valY)
    f.write(res)

    frames = np.arange(0, 900, 10)
    res = run("3 FPS", trainX[:,frames,:], trainY, testX[:,frames,:], testY, valX[:,frames,:], valY)
    f.write(res)

    frames = np.arange(0, 900, 30)
    res = run("1 FPS", trainX[:,frames,:], trainY, testX[:,frames,:], testY, valX[:,frames,:], valY)
    f.write(res)
