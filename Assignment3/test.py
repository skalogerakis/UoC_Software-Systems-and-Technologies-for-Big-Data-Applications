

# Keras supports two main types of model Sequential and Functional API(more advanced)
# https://keras.io/api/models/sequential/
# https://keras.io/guides/functional_api/

#Sequential -> linear stack of layers. Dense is the most common layer
import tensorflow as tf
from tensorflow import keras    #Keras high-level API of tensorflow
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.datasets import imdb

# plt.style.use('ggplot')

data = keras.datasets.imdb  #Load data from the existing dataset in keras

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=88000)

#Print the first one to see what is going on. Output is numbers.
# This is integer encoding words. Load integers in the list
print(train_data[0])

word_index = imdb.get_word_index()  #This gives tuples with string and word

word_index = {k:(v+3) for k, v in word_index.items()}   #Breaks into key-value. Start from three because we have special characters for word mapping
#Assign my values to the dictionary
word_index["<PAD>"] = 0 #Add zeros to make them the same length
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key,value) in word_index.items()]) #swap values and keys. We want integers to point to a word

#Data preprossing the easy way
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)


def decode_review(text):    #Return the human readable words
	return " ".join([reverse_word_index.get(i, "?") for i in text])


print(decode_review(test_data[0])) # Print the human readable sentence

print(len(test_data[0]),len(test_data[1]))  # Different length for each sentence. We need to make sure that all data have the same length. After padding everything should be the same

print("Dimension", train_data.shape[0]," Test ", train_data.shape[1])

#First effort
model = keras.Sequential()  # add layers of model
model.add(keras.layers.Dense(10,input_dim= train_data.shape[1],activation="relu"))        # This is actual neural network to perform classification
model.add(keras.layers.Dense(1, activation="sigmoid"))   # Takes any value and outputs between 0 and 1


#binary_crossentropy calculates the difference between real answer
model.compile(optimizer="adam",loss="binary_crossentropy", metrics=["accuracy"])	#Compile method configures learning process. Specify the optimizer and loss function
model.summary()

# Summary produces a result that we have 250,021 parameters
# 25000(each feature vector) * 10 (nodes). +10 times an added bias for each node. +10 weights +1 bias

# #validation data -> check how well our model is based on new data
#
x_val = train_data[:10000]  #Cut into 10000 entries
x_train = train_data[10000:]

y_val = train_labels[:10000]  #Cut into 10000 entries
y_train = train_labels[10000:]

# #Do not touch the test data, use to test our model. Use validation data to validate model
#
# #Batch check how many load at once
fitModel = model.fit(x_train,y_train, epochs=4, batch_size=10, validation_data=(x_val,y_val), verbose=1)
# #
# results = model.evaluate(test_data, test_labels, verbose=0)


loss, accuracy = model.evaluate(x_train, y_train, verbose=1)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(test_data, test_labels, verbose=1)
print("Testing Accuracy:  {:.4f}".format(accuracy))



# #
# print(results)

# model.save("model.h5")  # Save the model. h5 it is the extension in a model in keras. Transforms it in binary






# plot_history(fitModel)





embedding_dim = 16

model = keras.Sequential()
model.add(keras.layers.Embedding(input_dim=88000,   #Vocabulary size
                           output_dim=embedding_dim,
                           input_length=250))   #Length of input sequences.Required for fletten function
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

# #Batch check how many load at once
fitModel = model.fit(x_train,y_train, epochs=4, batch_size=512, validation_data=(x_val,y_val), verbose=1)
# #
# results = model.evaluate(test_data, test_labels, verbose=0)


loss, accuracy = model.evaluate(x_train, y_train, verbose=1)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(test_data, test_labels, verbose=1)
print("Testing Accuracy:  {:.4f}".format(accuracy))

# plot_history(fitModel)


embedding_dim = 16

model = keras.Sequential()
model.add(keras.layers.Embedding(input_dim=88000,
                           output_dim=embedding_dim,
                           input_length=250))
model.add(keras.layers.GlobalMaxPool1D())
model.add(keras.layers.Dense(10, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

loss, accuracy = model.evaluate(x_train, y_train, verbose=1)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(test_data, test_labels, verbose=1)
print("Testing Accuracy:  {:.4f}".format(accuracy))

# plot_history(fitModel)


embedding_dim = 16

model = keras.Sequential()
model.add(keras.layers.Embedding(88000, embedding_dim, input_length=250))
model.add(keras.layers.Conv1D(128, 5, activation='relu'))
model.add(keras.layers.GlobalMaxPooling1D())
model.add(keras.layers.Dense(10, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

loss, accuracy = model.evaluate(x_train, y_train, verbose=1)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(test_data, test_labels, verbose=1)
print("Testing Accuracy:  {:.4f}".format(accuracy))

# plot_history(fitModel)