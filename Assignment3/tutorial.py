import tensorflow as tf
from tensorflow import keras    #Keras high-level API of tensorflow
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.datasets import imdb

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

# define model
model = keras.Sequential()  # add layers of model
model.add(keras.layers.Embedding(88000,16))     # Create an embedding space. Words with the same meaning should be closer to one another. 16 is the number of coefficients
model.add(keras.layers.GlobalAveragePooling1D())    #This layer lowers the dimension of the problem. 16 dimension is a lot
model.add(keras.layers.Dense(16, activation="relu"))        # This is actual neural network to perform classification
model.add(keras.layers.Dense(1, activation="sigmoid"))   # Takes any value and outputs between 0 and 1

model.summary()

#binary_crossentropy calculates the difference between real answer
model.compile(optimizer="adam",loss="binary_crossentropy", metrics=["accuracy"])

#validation data -> check how well our model is based on new data

x_val = train_data[:10000]  #Cut into 10000 entries
x_train = train_data[10000:]

y_val = train_labels[:10000]  #Cut into 10000 entries
y_train = train_labels[10000:]

#Do not touch the test data, use to test our model. Use validation data to validate model

#Batch check how many load at once
fitModel = model.fit(x_train,y_train, epochs=5, batch_size=512, validation_data=(x_val,y_val),verbose=1)

results = model.evaluate(test_data, test_labels)

print(results)

model.save("model.h5")  # Save the model. h5 it is the extension in a model in keras. Transforms it in binary


# def review_encode(s):
# 	encoded = [1]
#
# 	for word in s:
# 		if word.lower() in word_index:
# 			encoded.append(word_index[word.lower()])
# 		else:
# 			encoded.append(2)
#
# 	return encoded
# model = keras.models.load_model("model.h5") #This is the way to load a existing model and don't run it all over again
#
#
# with open("externalReview.txt", encoding="utf-8") as f:
# 	for line in f.readlines():
# 		nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
# 		encode = review_encode(nline)
# 		encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250) # make the data 250 words long
# 		predict = model.predict(encode)
# 		print(line)
# 		print(encode)
# 		print(predict[0])

# Predict an example to see how well it behaves
# test_review = test_data[0]
# predict = model.predict(test_review)
# print("Review")
# print(decode_review(test_review))
# print("Prediction "+str(predict[0]))
# print("Actual "+str(test_labels[0]))
