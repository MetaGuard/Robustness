# Import libraries
import time
import gc
import numpy as np
import keras.backend as K
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras import layers
from lib.config import NUM_USERS, TRAIN_SIZE, VAL_SIZE, TEST_SIZE, BATCH_SIZE, LEARN_RATE

# Split an input dataset into training, validation, and testing sets
def split_data(data):
    # Split into training, validation, and testing sets
    trainX = data[:,:TRAIN_SIZE].reshape(-1, 900, 21)
    valX = data[:,TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE].reshape(-1, 900, 21)
    testX = data[:,TRAIN_SIZE+VAL_SIZE:].reshape(-1, 900, 21)

    # Create target class labels
    trainY = np.repeat(np.arange(NUM_USERS), TRAIN_SIZE)
    valY = np.repeat(np.arange(NUM_USERS), VAL_SIZE)
    testY = np.repeat(np.arange(NUM_USERS), TEST_SIZE)

    # Convert the labels to one-hot vectors
    trainY = tf.one_hot(trainY, depth=NUM_USERS)
    valY = tf.one_hot(valY, depth=NUM_USERS)
    testY = tf.one_hot(testY, depth=NUM_USERS)

    # Return training, validation, and testing sets
    return trainX, trainY, valX, valY, testX, testY

# Train and test an "LSTM Funnel" identification model with Keras
def run(name, trainX, trainY, testX, testY, valX, valY, N=NUM_USERS, T=TEST_SIZE, LR=LEARN_RATE):
    print("Now running:", name, "...")
    print("Training data shape:", trainX.shape)
    print("Validation data shape:", testX.shape)
    print("Testing data shape:", valX.shape)

    # Start measuring performance
    start_time = time.time()

    # Define the identification model architecture
    model = keras.Sequential([
        layers.LSTM(256, input_shape=(trainX.shape[-2], trainX.shape[-1]), return_sequences=True),
        layers.AveragePooling1D(pool_size=30),
        layers.LSTM(256),
        layers.Dense(256),
        layers.Dense(256),
        layers.Dense(N, activation="softmax")
    ], name='user-identification')

    # Configure the adam optimizer
    adam = keras.optimizers.Adam(learning_rate=LR)

    # Compile the model with loss and metrics
    model.compile(
        loss="categorical_crossentropy", # Use categorical crossentropy as the loss function
        optimizer=adam, # Use adam as the optimizer
        metrics=["accuracy"] # Use accuracy as the metric
    )

    # Configure early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=25,
        verbose=1,
        mode="max",
        restore_best_weights=True
    )

    # Train the model on the train data
    model.fit(trainX, trainY,
        epochs=500,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_data=(valX, valY),
        callbacks=[early_stopping]
    )

    # Save model
    model.save('./models/' + str(name) + '.keras')

    # Clear GPU memory
    K.clear_session()
    gc.collect()

    # Test per-sample accuracy
    predY = model.predict(testX, batch_size=BATCH_SIZE)
    realY = np.repeat(np.arange(N), T)
    sacc = accuracy_score(realY, predY.argmax(axis=1))
    print("Accuracy (Per Sample): " + str(sacc))

    # Test per-user accuracy
    with np.errstate(divide='ignore'):
        predY = np.log(predY).reshape(-1, T, N).sum(axis=1).argmax(axis=1)
    uacc = accuracy_score(list(range(N)), predY)
    print("Accuracy (Per User): " + str(uacc))

    # Print performance results
    end_time = time.time()
    elapsed = (end_time - start_time) / 60
    print("Finished in %s Minutes" % elapsed)

    # Clear GPU memory
    K.clear_session()
    gc.collect()

    # Return all results
    return name + "," + str(elapsed) + "," + str(sacc) + "," + str(uacc) + "\n"
