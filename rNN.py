from keras.models import Sequential as ks
from keras.layers import recurrent as kr

#Defs
model = ks.Sequential()
data = []

#Settings
filepath = ""
wfilepath = ""

def loadFile():
    global data
    data = open(filepath, "r")
    data = data.split()

loadFile():
model.add(kr.LSTM(1, activation = "sigmoid", recurrent_activation = "hard_sigmoid"))