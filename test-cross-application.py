# Import libraries
import numpy as np
from lib.model import run
import tensorflow as tf

# Load the input data
print("Loading VRNet data ...")
train = np.load('./data/vrnet-train.npy')
test = np.load('./data/vrnet-test.npy')

# Split into training, validation, and testing sets
trainX = train.reshape(-1, 900, 21)
testX = test.reshape(-1, 900, 21)

# Create target class labels
trainY = np.repeat(np.arange(5), 1000)
testY = np.repeat(np.arange(5), 1000)

# Convert the labels to one-hot vectors
trainY = tf.one_hot(trainY, depth=5)
testY = tf.one_hot(testY, depth=5)

# Test cross application
with open("./results/cross-application.csv", "w") as f:
    res = run("Cross Application", trainX, trainY, testX, testY, testX, testY, 5, 1000, 0.00001)
    f.write(res)
