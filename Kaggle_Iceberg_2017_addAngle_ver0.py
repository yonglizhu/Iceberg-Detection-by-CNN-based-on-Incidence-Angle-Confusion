# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 17:05:23 2017

@author: yzhu16
"""

import pandas as pd # Used to open CSV files 
import numpy as np # Used for matrix operations
import cv2 # Used for image augmentation
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,TensorBoard
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from sklearn import preprocessing
import pickle

np.random.seed(731)  #1337


df_train = pd.read_json('D:/FinalProj/train.json') 

log_path  = 'D:/FinalProj/kaggleuser_1/Log' # 

# Combining Angle information
tr_ang  = pickle.load( open("D:/FinalProj/tr_ang.pickle", "rb") )
temp=np.mean([0 if v == 'na' else v for v in tr_ang]) # Here, we use "mean" vlaue to fill up "na"
tr_ang = [temp if v == 'na' else v for v in tr_ang]  # can also try using "0" to fillup "na"
tr_ang = np.array(tr_ang).reshape(-1, 1)
tr_ang = np.deg2rad(tr_ang)


      
def get_scaled_imags(df,tr_ang):
    imgs = []
    
    for i, row in df.iterrows():
         
        ang = tr_ang[i]
        #make 75x75 image
        band_1 = np.array(row['band_1'])
        band_2 = np.array(row['band_2'])
        band_3 = band_1 + band_2 # plus since log(x*y) = log(x) + log(y)
        band_3 = band_3.reshape(75,75)
        
        band_1 = np.multiply(band_1, np.tile(np.cos(ang),(1,5625)))
        band_1 = band_1.reshape(75, 75)
        band_2 = np.multiply(band_2, np.tile(np.sin(ang),(1,5625)))
        band_2 = band_2.reshape(75, 75)

        # Rescale
        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())

        imgs.append(np.dstack((a, b, c)))

    return np.array(imgs)


Xtrain = get_scaled_imgs(df_train,tr_ang)
Ytrain = np.array(df_train['is_iceberg'])


df_train.inc_angle = df_train.inc_angle.replace('na',0) 
idx_tr = np.where(df_train.inc_angle>0)


Ytrain = Ytrain[idx_tr[0]]
Xtrain = Xtrain[idx_tr[0],...]


def get_more_images(imgs):
    
    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []
      
    for i in range(0,imgs.shape[0]):
        a=imgs[i,:,:,0]
        b=imgs[i,:,:,1]
        c=imgs[i,:,:,2]
        
        av=cv2.flip(a,1)
        ah=cv2.flip(a,0)
        bv=cv2.flip(b,1)
        bh=cv2.flip(b,0)
        cv=cv2.flip(c,1)
        ch=cv2.flip(c,0)
        
        vert_flip_imgs.append(np.dstack((av, bv, cv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch)))
      
    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)
       
    more_images = np.concatenate((imgs,v,h))
    
    return more_images

Xtr_more = get_more_images(Xtrain) 
Ytr_more = np.concatenate((Ytrain,Ytrain,Ytrain))

#X_valid_more = get_more_images(X_valid)
#y_valid_more = np.concatenate([y_valid, y_valid, y_valid])
#X_train, X_valid, y_train, y_valid = train_test_split(Xtrain, Ytrain, test_size=0.25) # old split ratio =0.25

def getModel():
    #Build keras model
    
    model=Sequential()
    
    # CNN 1
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.2))

    # CNN 2
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # CNN 3
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    #CNN 4
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    # You must flatten the data for the dense layers
    model.add(Flatten())

    #Dense 1
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))

    #Dense 2
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2)) 

    # Output 
    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(lr=0.001, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

model = getModel()
model.summary()


batch_size = 32
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
file_path="Wts-{epoch:02d}-{val_acc:.4f}-kaggleuser1_add_Angle.hdf5" # my added part
tb_cb = TensorBoard(log_dir=log_path, histogram_freq=0)

mcp_save = ModelCheckpoint(file_path, save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

callbacks=[earlyStopping, mcp_save, reduce_lr_loss, tb_cb]        
    
#------------------------------------------------------------------------------------------------------------------------------------------------------
history = model.fit(Xtr_more, Ytr_more, batch_size=batch_size, epochs=50, verbose=1,callbacks=callbacks, validation_split=0.25)

print(history.history.keys())
#
fig = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower left')
#
fig.savefig('performance_v2.png')
#---------------------------------------------------------------------------------------

# load the best model I got
model.load_weights(filepath = 'Wts-18-0.9257-kaggleuser1_add_Angle.hdf5')

score = model.evaluate(Xtrain, Ytrain, verbose=1)
print('Train score:', score[0])
print('Train accuracy:', score[1])


df_test = pd.read_json('D:\\FinalProj\\test.json')
#df_test.inc_angle = df_test.inc_angle.replace('na',0)

# Combining Angle information
te_ang  = pickle.load( open("D:/FinalProj/te_ang.pickle", "rb") )
temp=np.mean([0 if v == 'na' else v for v in te_ang]) # Here, we use "mean" vlaue to fill up "na"
te_ang = [temp if v == 'na' else v for v in te_ang]  # can also try using "0" to fillup "na"
te_ang = np.array(te_ang).reshape(-1, 1)
te_ang = np.deg2rad(te_ang)

Xtest = get_scaled_imags(df_test,te_ang)

#np.save('D:\\FinalProj\\Xtest', Xtest)
# Xtest = np.load('C:\\Users\\yzhu16\\.spyder-py3\\FinalDeep\\Xtest.npy')
pred_test = model.predict(Xtest)

submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})
print(submission.head(10))
submission.to_csv('submission_01062018_kaggleuser1.csv', index=False)
