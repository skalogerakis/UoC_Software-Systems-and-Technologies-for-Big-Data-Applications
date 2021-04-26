# Keras supports two main types of model Sequential and Functional API(more advanced)
# https://keras.io/api/models/sequential/
# https://keras.io/guides/functional_api/

# Sequential -> linear stack of layers. Dense is the most common layer
import tensorflow as tf
from tensorflow import keras  # Keras high-level API of tensorflow
import numpy as np
from tensorflow.python.keras.datasets import imdb
import matplotlib.pyplot as plt

vocab_size = 20000


def main():
    data = keras.datasets.imdb  # Load data from the existing dataset in keras

    # Only consider the top 20k words
    maxlen = 200  # Only consider the first 200 words of each movie review

    # https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb/load_data
    # Check put how load data works from imdb
    # Labeling is performed by sentiment (positive/negative)
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size,
                                                                          seed=100, oov_char=2)  # TODO change seed

    # Print the first one to see what is going on. Output is numbers.
    # This is integer encoding words. Load integers in the list
    # print("Train Data test :", train_data[0])
    # print("Train Label test :", train_labels[0])

    word_index = imdb.get_word_index()  # The word index dictionary. Keys are word strings, values are their index.

    # print(word_index)     # Prints the whole dictionary

    word_index = {k: (v + 3) for k, v in
                  word_index.items()}  # Breaks into key-value. Start from three because we have special characters for word mapping
    # Assign my values to the dictionary
    # https://stackoverflow.com/questions/42821330/restore-original-text-from-keras-s-imdb-dataset?rq=1
    word_index["<PAD>"] = 0  # Padding character. Add zeros to make them the same length
    word_index["<START>"] = 1   # The start of a sequence will be marked with this character
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3

    reverse_word_index = dict(
        zip(word_index.values(),
            word_index.keys()))  # Swap the position of key-value to make search easier.(search by key)

    # print(len(test_data[0]), len(test_data[1]))  # At a first glance sentences are of different length. Must change that and all should have the same length

    # Data preprocessing the easy way
    train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], dtype=np.uint8,
                                                            padding="post", maxlen=maxlen)
    test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], dtype=np.uint8,
                                                           padding="post", maxlen=maxlen)

    # print(len(test_data[0]), len(test_data[1]))  # Now all have the same length

    print(decode_review(test_data[0], reverse_word_index))  # Print the human readable sentence

    # print("Dimension", train_data.shape[0], " Test ", train_data.shape[1])

    # #validation data -> check how well our model is based on new data
    # Split dataset 80/20
    x_val = train_data[:4000]  # Cut into 10000 entries
    x_train = train_data[4000:]

    y_val = train_labels[:4000]  # Cut into 10000 entries
    y_train = train_labels[4000:]

    # All attempted models
    # A model starts overfitting when the loss of the validation data starts rising again.
    # Also when the number of epochs is too large for the models
    # initModel(maxlen, x_train, y_train, x_val, y_val, test_data, test_labels)
    # embeddingEffort(maxlen, x_train, y_train, x_val, y_val, test_data, test_labels)
    CNN(maxlen, x_train, y_train, x_val, y_val, test_data, test_labels)
    # LSTM(maxlen, x_train, y_train, x_val, y_val, test_data, test_labels)


def initModel(maxlen, x_train, y_train, x_val, y_val, test_data, test_labels):
    model = keras.Sequential()  # add layers of model
    # model.add(keras.layers.Dense(16, input_dim=maxlen,
    #                              activation="tanh"))  # Does not work well with tanh
    # model.add(keras.layers.Dense(10, input_dim=maxlen,
    #                              activation="relu"))  # Layer numbers dont seem to impact performance in this case
    model.add(keras.layers.Dense(16, input_dim=maxlen,
                                 activation="relu"))  # This is actual neural network to perform classification
    model.add(keras.layers.Dense(1, activation="sigmoid"))  # We need sigmoid function as it takes any value and outputs between 0 and 1

    # binary_crossentropy calculates the difference between real answer. Adam optimizer seems to be the best choice
    # model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
    # model.compile(optimizer="adadelta", loss="binary_crossentropy", metrics=["accuracy"])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"])
    # model.compile(optimizer="adamax", loss="binary_crossentropy", metrics=["accuracy"])

    # model.compile(optimizer="adam", loss="poisson", metrics=["accuracy"])   # Poisson loss does not perform well
    # model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"]) #Also a bad choice
    model.summary()

    # Summary produces a result that we have 2021 parameters
    # 200(each feature vector) * 10 (nodes). +10 times an added bias for each node. +10 weights +1 bias

    # Do not touch the test data, use to test our model. Use validation data to validate model
    # Batch check how many load at once
    # fitModel = model.fit(x_train, y_train, epochs=40, batch_size=100, validation_data=(x_val, y_val), verbose=1)  # Epoch change from 40 to 20 does not impact accuracy
    # fitModel = model.fit(x_train, y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val), verbose=1)  # The smaller the batch size the better the results. However, much slower!
    fitModel = model.fit(x_train, y_train, epochs=30, batch_size=100, validation_data=(x_val, y_val), verbose=1)

    evaluator(model, x_train, y_train, test_data, test_labels)
    # model.save("initSimple.h5")  # Save the model. h5 it is the extension in a model in keras. Transforms it in binary

    # model = keras.models.load_model("initSimple.h5") #This is the way to load a existing model and don't run it all over again

    plot_history(fitModel)


def embeddingEffort(maxlen, x_train, y_train, x_val, y_val, test_data, test_labels):
    embedding_dim = 16

    model = keras.Sequential()
    model.add(keras.layers.Embedding(input_dim=vocab_size,
                                     output_dim=embedding_dim,
                                     input_length=maxlen))  # Create an embedding space. Words with the same meaning should be closer to one another. 16 is the number of coefficients
    # model.add(keras.layers.GlobalMaxPooling1D())   # Pooling is a way to downsample the layer after embedding. This layer lowers the dimension of the problem.
    model.add(keras.layers.GlobalAveragePooling1D())    # The best pooling option so far
    # model.add(keras.layers.AveragePooling1D())  #The worst so far
    # model.add(keras.layers.MaxPooling1D())
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    fitModel = model.fit(x_train, y_train, epochs=40, batch_size=100, validation_data=(x_val, y_val), verbose=1)

    evaluator(model, x_train, y_train, test_data, test_labels)

    plot_history(fitModel)


def CNN(maxlen, x_train, y_train, x_val, y_val, test_data, test_labels):

    model = keras.Sequential()
    # model.add(keras.layers.Embedding(vocab_size, 16, input_length=maxlen))    # Reached 90% accuracy
    # model.add(keras.layers.Embedding(vocab_size, 32, input_length=maxlen))  # Increasing dimensions increases accuracy, however increases complexity and takes longer
    model.add(keras.layers.Embedding(vocab_size, 32, input_length=maxlen))  # Increasing dimensions increases accuracy, however increases complexity and takes longer
    # model.add(keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu'))
    # model.add(keras.layers.Conv1D(filters=16, kernel_size=5))
    # model.add(keras.layers.Conv1D(filters=16, kernel_size=5, activation='tanh'))
    model.add(keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu'))    # Generally speaking, relu function performs better than the others, as we can see from the results and

    # model.add(keras.layers.GlobalAveragePooling1D())  # Unlike the previous model, GlobalAverage performs  worse
    model.add(keras.layers.GlobalMaxPooling1D())
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()


    # fitModel = model.fit(x_train, y_train, epochs=10, batch_size=100, validation_data=(x_val, y_val), verbose=1)
    fitModel = model.fit(x_train, y_train, epochs=10, batch_size=50, validation_data=(x_val, y_val), verbose=1)

    evaluator(model, x_train, y_train, test_data, test_labels)

    plot_history(fitModel)


# LSTM takes too long in my setup to execute. Cannot properly evaluate results
# Adding layers does not help in gerenal in my setup
def LSTM(maxlen, x_train, y_train, x_val, y_val, test_data, test_labels):
    embedding_dim = 8
    dropout_val = 0.2
    lstm_val = 128

    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(keras.layers.Dropout(dropout_val))
    model.add(keras.layers.LSTM(lstm_val))
    model.add(keras.layers.Dropout(dropout_val))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    fitModel = model.fit(x_train, y_train, epochs=3, batch_size=512, validation_data=(x_val, y_val), verbose=1)

    evaluator(model, x_train, y_train, test_data, test_labels)

    plot_history(fitModel)


def evaluator(model, x_train, y_train, test_data, test_labels):
    results_train = model.evaluate(x_train, y_train, verbose=1)  # Evaluate returns loss and accuracy
    print("Train loss, train accuracy ", results_train)
    results_test = model.evaluate(test_data, test_labels, verbose=1)
    print("Test loss, Test accuracy ", results_test)


def decode_review(text, reverse_word_index):  # Return the human readable words
    return " ".join([reverse_word_index.get(i, "?") for i in text])


def plot_history(plot_model):
    # Fit model returns a history object. A history object is a record of training loss value and metrics value
    acc = plot_model.history['accuracy']
    val_acc = plot_model.history['val_accuracy']

    x = range(1, len(acc) + 1)

    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plt.plot(x, acc, 'b', label='Train accuracy')
    plt.plot(x, val_acc, 'r', label='Val accuracy')
    plt.title('Accuracy')
    plt.legend()

    loss = plot_model.history['loss']
    val_loss = plot_model.history['val_loss']
    plt.subplot(2, 1, 2)
    plt.plot(x, loss, 'b', label='Train loss')
    plt.plot(x, val_loss, 'r', label='Val loss')
    plt.title('Loss')
    plt.legend()

    # Show everything
    plt.show()


if __name__ == "__main__":
    main()
