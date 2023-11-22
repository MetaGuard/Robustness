# Import libraries
import gc
import numpy as np
from lib.model import run, split_data

# Load the input data
print("Loading Beat Saber data ...")
data = np.load('./data/boxrr.npy')
trainX, trainY, valX, valY, testX, testY = split_data(data)

# Test added noise
with open("./results/added-noise.csv", "w") as f:
    for sd in [0.1, 0.5, 1.0, 2.0, 5.0]:
        train_data = trainX + np.random.normal(loc=0.0, scale=sd, size=trainX.shape).astype('float16')
        test_data = testX + np.random.normal(loc=0.0, scale=sd, size=testX.shape).astype('float16')
        val_data = valX + np.random.normal(loc=0.0, scale=sd, size=valX.shape).astype('float16')
        res = run("Noise sd=" + str(sd), train_data.astype('float16'), trainY, test_data.astype('float16'), testY, val_data.astype('float16'), valY)
        f.write(res)
        gc.collect()
