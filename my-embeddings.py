from string import punctuation
from os import listdir
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import csv
from sklearn.model_selection import train_test_split
import pandas as pd

def load_docs(filename, vocab):
	text = list()
	label = list()
	with open(filename) as f:
		reader = csv.reader(f)
		for row in reader:
			if row[3] == "unsup":
				break;
			else:
				text.append(clean_doc(row[2],vocab))
				if row[3]=="neg":
					label.append(0)
				else :
					label.append(1)
	return text,label

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r',encoding="cp1252")
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# turn a doc into clean tokens
def clean_doc(doc, vocab):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# filter out tokens not in vocab
	tokens = [w for w in tokens if w in vocab]
	tokens = ' '.join(tokens)
	return tokens

# load embedding as a dict
def load_embedding(filename):
	# load embedding into memory, skip first line
	file = open(filename,'r',encoding="UTF8")
	lines = file.readlines()
	file.close()
	# create a map of words to vectors
	embedding = dict()
	for line in lines:
		parts = line.split()
		# key is string word, value is numpy array for vector
		embedding[parts[0]] = np.asarray(parts[1:], dtype='float32')
	return embedding

# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab):
	# total vocabulary size plus 0 for unknown words
	vocab_size = len(vocab) + 1
	# define weight matrix dimensions with all 0
	weight_matrix = np.zeros((vocab_size, embeddingSize))
	# step vocab, store vectors using the Tokenizer's integer mapping
	for word, i in vocab.items():
		vector = embedding.get(word)
		if vector is not None:
			weight_matrix[i] = vector
	return weight_matrix

#size of the embedding that will be used 
embeddingSize=50
# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# load all training reviews

[all_docs,labels] = load_docs('goodDf.csv',vocab)
# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(all_docs)

# sequence encode
encoded_docs = tokenizer.texts_to_sequences(all_docs)
# pad sequences
max_length = max([len(s.split()) for s in all_docs])
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(len(padded_docs))
Xtrain,Xtest,Ytrain,Ytest=train_test_split(padded_docs,labels,test_size=0.1)

# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1

embedding_layer = Embedding(vocab_size, embeddingSize, input_length=max_length)

# define model
nn = Sequential()
nn.add(embedding_layer)
nn.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
nn.add(MaxPooling1D(pool_size=2))
nn.add(Flatten())
nn.add(Dropout(0.4))
nn.add(Dense(64, activation='relu'))
nn.add(Dense(1, activation='sigmoid'))
print(nn.summary())
# compile network
nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'],)
# fit network
history = nn.fit(Xtrain, Ytrain, epochs=6, validation_split=0.10)

##Store Plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#Accuracy plot
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
#No validation loss in this example
plt.legend(['train','val'], loc='upper left')
plt.savefig('output_code3/model_accuracy.pdf')
plt.close()
#Loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig('output_code3/model_loss.pdf')

#Confusion Matrix
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
#Compute probabilities
pred = nn.predict(Xtest)
y_pred=list()
for i in range(0,len(pred)):
	if(pred[i]<=0.5):
		y_pred.append(0)
	else:
		y_pred.append(1)
	
	
#Plot statistics
print( 'Analysis of results')
target_names = ['neg', 'pos']
print(classification_report(Ytest, y_pred,target_names=target_names))
print(confusion_matrix(Ytest, y_pred))

# evaluate
loss, acc = nn.evaluate(Xtest, Ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))
