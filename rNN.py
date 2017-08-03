import numpy as np
import keras.utils as npUtils
from keras.models import Sequential as ks
from keras.layers import recurrent as kr
from keras.layers import Dropout as kd
from keras.layers import Dense as kD
from keras.callbacks import ModelCheckpoint as mc

#Defs
dataX = []
dataY = []

#Settings
filepath = "hamlet.txt"
wfilepath = "weights.hdf5"
trainLen = 100

#This part is following a tutorial from <https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/>
rawdata = open(filepath, "r").read()
rawdata = rawdata.lower()
chars = sorted(list(set(rawdata)))
charToInt = dict((c, i) for i, c in enumerate(chars))
nChars = len(rawdata)
nVocab = len(chars)
print(nVocab)

for i in range(nChars - trainLen):
    seqIn = rawdata[i:i + trainLen]
    seqOut = rawdata[i + trainLen]
    dataX.append([charToInt[char] for char in seqIn])
    dataY.append(charToInt[seqOut])

nPatterns = len(dataX)
print(nPatterns)

X = np.reshape(dataX, (nPatterns, trainLen, 1))
X = X / float(nVocab)

y = npUtils.to_categorical(dataY)

model = ks()
model.add(kr.LSTM(256, input_shape = (X.shape[1], X.shape[2])))
model.add(kd(0.2))
model.add(kD(y.shape[1], activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

checkpoint = mc(wfilepath, monitor = 'loss', verbose = 1, save_best_only = True)
callbacksList = [checkpoint]

#Here we go!
model.fit(X, y, epochs = 20, batch_size = 128, callbacks = callbacksList)