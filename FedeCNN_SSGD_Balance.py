# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 14:42:58 2020

@author: Rain
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from keras.models import Model
from keras.layers import Input,Flatten, Dense, BatchNormalization, Dropout, Conv1D,Activation
from keras.callbacks import TensorBoard, ModelCheckpoint
import scipy.io as scio
import numpy as np
from numpy import array
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
L = 128 #sample points

feature1 = Input(shape=[L, 2], name='CNN1')
feature2 = Input(shape=[L, 2], name='CNN2')
feature3 = Input(shape=[L, 2], name='CNN3')
feature4 = Input(shape=[L, 2], name='CNN4')

Shared_Conv1 = Conv1D(128, 16, activation='relu', padding='same')
Shared_BN1 = BatchNormalization()
Shared_Dr1 = Dropout(0.1)
Shared_Conv2= Conv1D(64, 8, activation='relu', padding='same')
Shared_BN2 = BatchNormalization()
Shared_Dr2 = Dropout(0.1)
Shared_Flatten = Flatten()
Shared_Dense1 = Dense(256, activation='relu')
Shared_BN3 = BatchNormalization()
Shared_Dr3 = Dropout(0.5)
Shared_Dense2 = Dense(128, activation='relu')
Shared_BN4 = BatchNormalization()
Shared_Dr4 = Dropout(0.5)
Shared_Dense3 = Dense(4)

classifier1 = Shared_Conv1(feature1)
classifier1 = Shared_BN1(classifier1)
classifier1 = Shared_Dr1(classifier1)
classifier1 = Shared_Conv2(classifier1)
classifier1 = Shared_BN2(classifier1)
classifier1 = Shared_Dr2(classifier1)
classifier1 = Shared_Flatten(classifier1)
classifier1 = Shared_Dense1(classifier1)
classifier1 = Shared_BN3(classifier1)
classifier1 = Shared_Dr3(classifier1)
classifier1 = Shared_Dense2(classifier1)
classifier1 = Shared_BN4(classifier1)
classifier1 = Shared_Dr4 (classifier1)
classifier1 = Shared_Dense3 (classifier1)
classifier1 = Activation('softmax', name='Classifier1')(classifier1)

classifier2 = Shared_Conv1(feature2)
classifier2 = Shared_BN1(classifier2)
classifier2 = Shared_Dr1(classifier2)
classifier2 = Shared_Conv2(classifier2)
classifier2 = Shared_BN2(classifier2)
classifier2 = Shared_Dr2(classifier2)
classifier2 = Shared_Flatten(classifier2)
classifier2 = Shared_Dense1(classifier2)
classifier2 = Shared_BN3(classifier2)
classifier2 = Shared_Dr3(classifier2)
classifier2 = Shared_Dense2(classifier2)
classifier2 = Shared_BN4(classifier2)
classifier2 = Shared_Dr4 (classifier2)
classifier2 = Shared_Dense3 (classifier2)
classifier2 = Activation('softmax', name='Classifier2')(classifier2)

classifier3 = Shared_Conv1(feature3)
classifier3 = Shared_BN1(classifier3)
classifier3 = Shared_Dr1(classifier3)
classifier3 = Shared_Conv2(classifier3)
classifier3 = Shared_BN2(classifier3)
classifier3 = Shared_Dr2(classifier3)
classifier3 = Shared_Flatten(classifier3)
classifier3 = Shared_Dense1(classifier3)
classifier3 = Shared_BN3(classifier3)
classifier3 = Shared_Dr3(classifier3)
classifier3 = Shared_Dense2(classifier3)
classifier3 = Shared_BN4(classifier3)
classifier3 = Shared_Dr4 (classifier3)
classifier3 = Shared_Dense3 (classifier3)
classifier3 = Activation('softmax', name='Classifier3')(classifier3)

classifier4 = Shared_Conv1(feature4)
classifier4 = Shared_BN1(classifier4)
classifier4 = Shared_Dr1(classifier4)
classifier4 = Shared_Conv2(classifier4)
classifier4 = Shared_BN2(classifier4)
classifier4 = Shared_Dr2(classifier4)
classifier4 = Shared_Flatten(classifier4)
classifier4 = Shared_Dense1(classifier4)
classifier4 = Shared_BN3(classifier4)
classifier4 = Shared_Dr3(classifier4)
classifier4 = Shared_Dense2(classifier4)
classifier4 = Shared_BN4(classifier4)
classifier4 = Shared_Dr4 (classifier4)
classifier4 = Shared_Dense3 (classifier4)
classifier4 = Activation('softmax', name='Classifier4')(classifier4)

model = Model(inputs = [feature1,feature2,feature3,feature4], outputs=[classifier1,classifier2,classifier3,classifier4])
model.compile(optimizer='sgd',
              loss={'Classifier1':'categorical_crossentropy',
                    'Classifier2':'categorical_crossentropy',
                    'Classifier3':'categorical_crossentropy',
                    'Classifier4':'categorical_crossentropy'},
              loss_weights={'Classifier1':0.25,
                            'Classifier2':0.25,
                            'Classifier3':0.25,
                            'Classifier4':0.25})
model.summary()

N_Large = 12000
N_Small = 2000
N_Val = 1000

#####train
data = scio.loadmat("train/part1.mat")
X = data.get('IQ')
X_BPSK = X[0:N_Large,:,:]

X_QPSK = X[N_Large:N_Large+N_Large,:,:]

X_8PSK = X[N_Large+N_Large:N_Large+N_Large+N_Small,:,:]
X_8PSK = np.vstack((X_8PSK,X_8PSK,X_8PSK,X_8PSK,X_8PSK,X_8PSK)) 

X_16QAM = X[N_Large+N_Large+N_Small:N_Large+N_Large+N_Small+N_Small,:,:]
X_16QAM = np.vstack((X_16QAM,X_16QAM,X_16QAM,X_16QAM,X_16QAM,X_16QAM)) 

X_train1 = np.vstack((X_BPSK,X_QPSK,X_8PSK,X_16QAM))
y11 = np.zeros([N_Large,1])
y12 = np.ones([N_Large,1])
y13 = np.ones([N_Large,1])*2
y14 = np.ones([N_Large,1])*3
y1 = np.vstack((y11,y12,y13,y14))
y1 = array(y1)
Y_train1 = to_categorical(y1)

data = scio.loadmat("train/part2.mat")
X = data.get('IQ')
X_BPSK = X[0:N_Small,:,:]
X_BPSK = np.vstack((X_BPSK,X_BPSK,X_BPSK,X_BPSK,X_BPSK,X_BPSK))

X_QPSK = X[N_Small:N_Small+N_Large,:,:]

X_8PSK = X[N_Small+N_Large:N_Small+N_Large+N_Large,:,:]

X_16QAM = X[N_Small+N_Large+N_Large:N_Small+N_Large+N_Large+N_Small,:,:]
X_16QAM = np.vstack((X_16QAM,X_16QAM,X_16QAM,X_16QAM,X_16QAM,X_16QAM)) 

X_train2 = np.vstack((X_BPSK,X_QPSK,X_8PSK,X_16QAM))
y21 = np.zeros([N_Large,1])
y22 = np.ones([N_Large,1])
y23 = np.ones([N_Large,1])*2
y24 = np.ones([N_Large,1])*3
y2 = np.vstack((y21,y22,y23,y24))
y2 = array(y2)
Y_train2 = to_categorical(y2)

data = scio.loadmat("train/part3.mat")
X = data.get('IQ')

X_BPSK = X[0:N_Small,:,:]
X_BPSK = np.vstack((X_BPSK,X_BPSK,X_BPSK,X_BPSK,X_BPSK,X_BPSK)) 

X_QPSK = X[N_Small:N_Small+N_Small,:,:]
X_QPSK = np.vstack((X_QPSK,X_QPSK,X_QPSK,X_QPSK,X_QPSK,X_QPSK)) 

X_8PSK = X[N_Small+N_Small:N_Small+N_Small+N_Large,:,:]

X_16QAM = X[N_Small+N_Small+N_Large:N_Small+N_Small+N_Large+N_Large,:,:]

X_train3 = np.vstack((X_BPSK,X_QPSK,X_8PSK,X_16QAM))
y31 = np.zeros([N_Large,1])
y32 = np.ones([N_Large,1])
y33 = np.ones([N_Large,1])*2
y34 = np.ones([N_Large,1])*3
y3 = np.vstack((y31,y32,y33,y34))
y3 = array(y3)
Y_train3 = to_categorical(y3)

data = scio.loadmat("train/part4.mat")
X = data.get('IQ')

X_BPSK = X[0:N_Large,:,:]

X_QPSK = X[N_Large:N_Large+N_Small,:,:]
X_QPSK = np.vstack((X_QPSK,X_QPSK,X_QPSK,X_QPSK,X_QPSK,X_QPSK)) 

X_8PSK = X[N_Large+N_Small:N_Large+N_Small+N_Small,:,:]
X_8PSK = np.vstack((X_8PSK,X_8PSK,X_8PSK,X_8PSK,X_8PSK,X_8PSK)) 

X_16QAM = X[N_Large+N_Small+N_Small:N_Large+N_Small+N_Small+N_Large,:,:]

X_train4 = np.vstack((X_BPSK,X_QPSK,X_8PSK,X_16QAM))
y41 = np.zeros([N_Large,1])
y42 = np.ones([N_Large,1])
y43 = np.ones([N_Large,1])*2
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

checkpoint = ModelCheckpoint("FedeCNN_SSGD_Balance.hdf5", 
                      verbose=1, 
                      save_best_only=True)
tensorboard = TensorBoard("FedeCNN_SSGD_Balance.log", 0)
model.fit([X_train1, X_train2,X_train3, X_train4],
       [Y_train1, Y_train2,Y_train3, Y_train4],
       batch_size=100,
       epochs=1000,
       validation_data=([X_val1, X_val2,X_val3, X_val4],
                        [Y_val1, Y_val2,Y_val3, Y_val4]),
       callbacks=[checkpoint, tensorboard])

model.load_weights("FedeCNN_SSGD_Balance.hdf5")
snrs = range(-10, 12, 2)
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
	[A1,A2,A3,A4] = model.predict([x, x, x, x], batch_size = 100, verbose=2)
	pre_lables = []
	for j in A1:
	    tmp = np.argmax(j, 0)
	    pre_lables.append(tmp)
	cm = confusion_matrix(y, pre_lables)
	acc = (cm[0,0] + cm[1,1] + cm[2,2] + cm[3,3])/(N*4)
	print(acc)