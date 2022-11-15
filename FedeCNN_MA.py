# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 14:42:58 2020

@author: Rain
"""

from keras import models
from keras.layers import Flatten, Dense, BatchNormalization, Dropout, Conv1D
def CNN():
	L = 128 #sample points
	model = models.Sequential()
	model.add(Conv1D(128, 16, activation='relu', padding='same',input_shape=[L, 2]))
	model.add(BatchNormalization())
	model.add(Dropout(0.1))

	model.add(Conv1D(64, 8, activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Dropout(0.1))

	model.add(Flatten())

	model.add(Dense(256, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))

	model.add(Dense(128, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))

	model.add(Dense(4, activation='softmax'))

	model.compile(loss='categorical_crossentropy',
	              optimizer='sgd', 
	              metrics=['accuracy'])
	return model
import keras
def averagemodel(weightpath, model):
	weights = []
	new_weights = list()
	for i in range(len(weightpath)):
	    model.load_weights(weightpath[i])
	    weight = model.get_weights()
	    weights.append(weight)

	for weights_list_tuple in zip(*weights):
	        new_weights.append([np.array(weights_).mean(axis=0) \
	             for weights_ in zip(*weights_list_tuple)])
	model.set_weights(new_weights)
	model.save_weights('MA.hdf5')
	return model

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
 
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

import scipy.io as scio
import numpy as np
from numpy import array
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

N_Large = 12000
N_Small = 2000
N_Val = 1000

#####train
data = scio.loadmat("train/part1.mat")
X = data.get('IQ')
X_train1 = X
y11 = np.zeros([N_Large,1])
y12 = np.ones([N_Large,1])
y13 = np.ones([N_Small,1])*2
y14 = np.ones([N_Small,1])*3
y1 = np.vstack((y11,y12,y13,y14))
y1 = array(y1)
Y_train1 = to_categorical(y1)

data = scio.loadmat("train/part2.mat")
X = data.get('IQ')
X_train2 = X
y21 = np.zeros([N_Small,1])
y22 = np.ones([N_Large,1])
y23 = np.ones([N_Large,1])*2
y24 = np.ones([N_Small,1])*3
y2 = np.vstack((y21,y22,y23,y24))
y2 = array(y2)
Y_train2 = to_categorical(y2)

data = scio.loadmat("train/part3.mat")
X = data.get('IQ')
X_train3 = X
y31 = np.zeros([N_Small,1])
y32 = np.ones([N_Small,1])
y33 = np.ones([N_Large,1])*2
y34 = np.ones([N_Large,1])*3
y3 = np.vstack((y31,y32,y33,y34))
y3 = array(y3)
Y_train3 = to_categorical(y3)

data = scio.loadmat("train/part4.mat")
X = data.get('IQ')
X_train4 = X
y41 = np.zeros([N_Large,1])
y42 = np.ones([N_Small,1])
y43 = np.ones([N_Small,1])*2
y44 = np.ones([N_Large,1])*3
y4 = np.vstack((y41,y42,y43,y44))
y4 = array(y4)
Y_train4 = to_categorical(y4)
###val
data = scio.loadmat("val/part1.mat")
X_val1 = data.get('IQ')
y11 = np.zeros([N_Val,1])
y12 = np.ones([N_Val,1])
y13 = np.ones([N_Val,1])*2
y14 = np.ones([N_Val,1])*3
y1 = np.vstack((y11,y12,y13,y14))
y1 = array(y1)
Y_val1 = to_categorical(y1)

data = scio.loadmat("val/part2.mat")
X_val2 = data.get('IQ')
y21 = np.zeros([N_Val,1])
y22 = np.ones([N_Val,1])
y23 = np.ones([N_Val,1])*2
y24 = np.ones([N_Val,1])*3
y2 = np.vstack((y21,y22,y23,y24))
y2 = array(y2)
Y_val2 = to_categorical(y2)

data = scio.loadmat("val/part3.mat")
X_val3 = data.get('IQ')
y31 = np.zeros([N_Val,1])
y32 = np.ones([N_Val,1])
y33 = np.ones([N_Val,1])*2
y34 = np.ones([N_Val,1])*3
y3 = np.vstack((y31,y32,y33,y34))
y3 = array(y3)
Y_val3 = to_categorical(y3)

data = scio.loadmat("val/part4.mat")
X_val4 = data.get('IQ')
y41 = np.zeros([N_Val,1])
y42 = np.ones([N_Val,1])
y43 = np.ones([N_Val,1])*2
y44 = np.ones([N_Val,1])*3
y4 = np.vstack((y41,y42,y43,y44))
y4 = array(y4)
Y_val4 = to_categorical(y4)

X_val = np.vstack((X_val1, X_val2, X_val3, X_val4))
Y_val = np.vstack((Y_val1, Y_val2, Y_val3, Y_val4))

sub_epoch = 1
epochs = int(10/sub_epoch)
batch_size = 100
start_epoch = -1
model = CNN()
loss1 = []
loss2 = []
loss3 = []
loss4 = []
weightpath = ['Sub1.hdf5','Sub2.hdf5','Sub3.hdf5','Sub4.hdf5']
model = CNN()
for epoch in range(epochs):
	# SubModel1
	if epoch !=start_epoch:
		model.load_weights("MA.hdf5")
	checkpoint = ModelCheckpoint(weightpath[0], 
	                       verbose=1, 
	                       save_best_only=True)
	history = LossHistory()
	model.fit(X_train1,Y_train1,
	        batch_size=batch_size,
	        epochs=sub_epoch,
	        validation_data=(X_val1,Y_val1),
	        callbacks=[checkpoint, history])	
	loss1.append(np.mean(history.losses))

	# SubModel2
	if epoch !=start_epoch:
		model.load_weights("MA.hdf5")
	checkpoint = ModelCheckpoint(weightpath[1], 
	                       verbose=1, 
	                       save_best_only=True)
	history = LossHistory()
	model.fit(X_train2,Y_train2,
	        batch_size=batch_size,
	        epochs=sub_epoch,
	        validation_data=(X_val2,Y_val2),
	        callbacks=[checkpoint, history])
	loss2.append(np.mean(history.losses))

	# SubModel3
	if epoch !=start_epoch:
		model.load_weights("MA.hdf5")
	checkpoint = ModelCheckpoint(weightpath[2], 
	                       verbose=1, 
	                       save_best_only=True)
	history = LossHistory()    
	model.fit(X_train3,Y_train3,
	        batch_size=batch_size,
	        epochs=sub_epoch,
	        validation_data=(X_val3,Y_val3),
	        callbacks=[checkpoint, history])
	loss3.append(np.mean(history.losses))

	# SubModel4
	if epoch !=start_epoch:
		model.load_weights("MA.hdf5")
	checkpoint = ModelCheckpoint(weightpath[3], 
	                       verbose=1, 
	                       save_best_only=True)
	history = LossHistory()
	model.fit(X_train4,Y_train4,
	        batch_size=batch_size,
	        epochs=sub_epoch,
	        validation_data=(X_val4,Y_val4),
	        callbacks=[checkpoint, history])
	loss4.append(np.mean(history.losses))
	
	averagemodel(weightpath, model)
	model.load_weights("MA.hdf5")

	[loss, acc]=model.evaluate(X_val,Y_val, batch_size=batch_size, verbose=1)
	print('The '+str(epoch+1)+' epoch: val_loss='+str(loss)+', val_acc='+str(acc))

import pandas as pd
df = pd.DataFrame(loss1, columns=['loss1'])
df.to_excel("loss1.xlsx", index=False)
df = pd.DataFrame(loss2, columns=['loss2'])
df.to_excel("loss2.xlsx", index=False)
df = pd.DataFrame(loss3, columns=['loss3'])
df.to_excel("loss3.xlsx", index=False)
df = pd.DataFrame(loss4, columns=['loss4'])
df.to_excel("loss4.xlsx", index=False)

snrs = range(-10, 12, 2)
model = CNN()
model.load_weights("MA.hdf5")
for snr in snrs:
   data_path="test/snr="+str(snr)+".mat"
   data = scio.loadmat(data_path)
   x = data.get('IQ')
   N = 20000
   y1 = np.zeros([N,1])
   y2 = np.ones([N,1])
   y3 = np.ones([N,1])*2
   y4 = np.ones([N,1])*3
   y = np.vstack((y1,y2,y3,y4))
   y = array(y)
   y = to_categorical(y)
   X_test=x
   Y_test=y
   [loss, acc] = model.evaluate(X_test,Y_test, batch_size=1000, verbose=0)
   print(acc)