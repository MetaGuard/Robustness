# Import libraries
import numpy as np
from lib.model import run, split_data

# Load the input data
print("Loading Beat Saber data ...")
data = np.load('./data/boxrr.npy')
trainX, trainY, valX, valY, testX, testY = split_data(data)

# Test dimensionality reductions
with open("./results/reduced-dimensions.csv", "w") as f:
    res = run("All features", trainX, trainY, testX, testY, valX, valY)
    f.write(res)

    res = run("Only hands", trainX[:,:,7:], trainY, testX[:,:,7:], testY, valX[:,:,7:], valY)
    f.write(res)

    ax = [10, 11, 12, 13, 17, 18, 19, 20]
    res = run("Only hand rotations", trainX[:,:,ax], trainY, testX[:,:,ax], testY, valX[:,:,ax], valY)
    f.write(res)

    ax = [10, 11, 12, 13]
    res = run("Only left hand rotations", trainX[:,:,ax], trainY, testX[:,:,ax], testY, valX[:,:,ax], valY)
    f.write(res)

    res = run("Only left hand rotational magnitude", trainX[:,:,[13]], trainY, testX[:,:,[13]], testY, valX[:,:,[13]], valY)
    f.write(res)
