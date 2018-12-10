# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 13:00:12 2017

@author: yzhu16
"""


from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing
import numpy as np
import json

path = 'D://FinalProj//'

# read raw data
tr = json.load(open('train.json'))

tr_band_1 =[]
tr_band_2 =[]
tr_label = []
tr_id = []
tr_ang =[]
for i, dict_ in enumerate(tr):
    tr_band_1.append(dict_['band_1'])
    tr_band_2.append(dict_['band_2'])
    tr_label.append(dict_['is_iceberg'])
    tr_id.append(dict_['id'])
    tr_ang.append(dict_['inc_angle'])

tr_band_1 = np.array(tr_band_1)
tr_band_2 = np.array(tr_band_2)
tr_label = np.array(tr_label).reshape(len(tr_label),1)
 
#pickle.dump( tr_ang, open( "tr_ang.pickle", "wb" ) )
#tr_ang2 = pickle.load(open( "tr_ang.pickle", "rb" ) )

min_max_scaler = preprocessing.MinMaxScaler()
tr_band_1 = min_max_scaler.fit_transform(tr_band_1)
min_max_scaler = preprocessing.MinMaxScaler()
tr_band_2 = min_max_scaler.fit_transform(tr_band_2)
tr_label_ =  np.hstack((tr_label, 1 - tr_label)) # manually one-hot for 2-classes


# ----------------Single-band experiemtns---------------------------------
# Training and Validation set deviding
tr_len = int(2/3*len(tr_label))
Xtr = tr_band_1[0: tr_len,:]
Xtr = Xtr.reshape((len(Xtr),75,75,1))

# Validattion set
Xval = tr_band_1[tr_len:,:]
Xval =Xval.reshape((len(Xval),75,75,1))

np.save(path+'Xtr_band1', Xtr)
np.save(path+'Xval_band1', Xval)


Xtr = tr_band_2[0: tr_len,:].reshape((len(Xtr),75,75,1))
Xval = tr_band_2[tr_len:,:].reshape((len(Xval),75,75,1))
np.save(path+'Xtr_band2', Xtr)
np.save(path+'Xval_band2', Xval)


Ytr = tr_label_[0: tr_len,:]
Yval = tr_label_[tr_len:,:]
np.save(path+'Ytr', Ytr)
np.save(path+'Yval', Yval)



#----------------Dual-Band experiemetn
tr_len = int(2/3*len(tr_label))


Xtr = tr_band_1[0: tr_len,:]
Xtr = np.concatenate((Xtr,tr_band_2[0: tr_len,:]), axis=1)
Xtr = Xtr.reshape((len(Xtr),75,75,2))

# Validattion set
Xval = tr_band_1[tr_len:,:]
Xval = np.concatenate((Xval, tr_band_2[tr_len:,:]), axis=1)
Xval = Xval.reshape((len(Xval),75,75,2))

np.save(path+'Xtr_dual', Xtr)
np.save(path+'Xval_dual', Xval)

## Add incidence anggle: way-1:
#temp=np.mean([0 if v == 'na' else v for v in tr_ang])
#tr_ang = [temp if v == 'na' else v for v in tr_ang]
#min_max_scaler = preprocessing.MinMaxScaler()
#tr_ang = min_max_scaler.fit_transform(tr_ang)
#tr_ang =tr_ang.reshape(-1,1)
#tr_len = int(2/3*len(tr_label))
#Xtr = tr_band_1[0: tr_len,:]
#Xtr = np.concatenate((Xtr,tr_band_2[0: tr_len,:],tr_ang[0: tr_len,:]), axis=1)
#Xtr = Xtr.reshape((len(Xtr),58,97,2))
#
#Xval = tr_band_1[tr_len:,:]
#Xval = np.concatenate((Xval, tr_band_2[tr_len:,:],tr_ang[tr_len:,:]), axis=1)
#Xval = Xval.reshape((len(Xval),58,97,2))


# Add incidence anggle: way-2: 
'''
 Xnew= Band_1*cos(phi)+Band_2*sin(phi)  Dec.12.2017
 '''
temp=np.mean([0 if v == 'na' else v for v in tr_ang])
tr_ang = [temp if v == 'na' else v for v in tr_ang]
min_max_scaler = preprocessing.MinMaxScaler()
tr_ang = min_max_scaler.fit_transform(tr_ang)
tr_ang =tr_ang.reshape(-1,1)
tr_len = int(2/3*len(tr_label))

Xtr = tr_band_1[0: tr_len,:]
Xtr = Xtr*np.tile(np.cos(tr_ang[0: tr_len,:]),(1,5625))
Xtr_ = tr_band_2[0: tr_len,:]
Xtr_ = Xtr_*np.tile(np.sin(tr_ang[0: tr_len,:]),(1,5625))
Xtr = Xtr + Xtr_
Xtr = Xtr.reshape((len(Xtr),75,75,1))
np.save(path+'Xtr_dual_ang', Xtr)

Xval = tr_band_1[tr_len:,:]
Xval = Xval*np.tile(np.cos(tr_ang[tr_len:,:]),(1,5625))
Xval_ = tr_band_2[tr_len:,:]
Xval_ = Xval_*np.tile(np.sin(tr_ang[tr_len:,:]),(1,5625))
Xval = Xval + Xval_
Xval = Xval.reshape((len(Xval),75,75,1))
np.save(path+'Xval_dual_ang', Xval)






# ------------------------------------------------------------------------#
# Read testing data
te = json.load(open('test.json'))
te_band_1 =[]
te_band_2 =[]
te_id = []
te_ang = []
for i, dict_ in enumerate(te):
    te_band_1.append(dict_['band_1'])
    te_band_2.append(dict_['band_2'])
    te_id.append(dict_['id'])
    te_ang.append(dict_['inc_angle'])

#pickle.dump( te_ang, open( "te_ang.pickle", "wb" ) )
 
np.save(path+'te_id', te_id)

te_band_1 = np.array(te_band_1)
te_band_2 = np.array(te_band_2) 

# Single-band test band-1:
min_max_scaler2 = preprocessing.MinMaxScaler()
Xte = min_max_scaler2.fit_transform(te_band_1).reshape(len(te_band_1), 75,75,1)
Xte.shape
np.save(path+'Xte_band1', Xte)

# Single-band test band-2:
min_max_scaler2 = preprocessing.MinMaxScaler()
Xte = min_max_scaler2.fit_transform(te_band_2).reshape(len(te_band_2), 75,75,1)
Xte.shape
np.save(path+'Xte_band2', Xte)


# Dual-band test No angle!:
min_max_scaler = preprocessing.MinMaxScaler()
te_band_1 = min_max_scaler.fit_transform(te_band_1)
min_max_scaler = preprocessing.MinMaxScaler()
te_band_2 = min_max_scaler.fit_transform(te_band_2)

Xte = np.concatenate((te_band_1,te_band_2), axis=1)
Xte = Xte.reshape((len(Xte),75,75,2))

Xte.shape
np.save(path+'Xte_dual', Xte)


# Add incidence anggl + Dual Band: way-2:
'''
 Xnew= Band_1*cos(phi)+Band_2*sin(phi)  Dec.12.2017
 '''
 
min_max_scaler = preprocessing.MinMaxScaler()
te_band_1 = min_max_scaler.fit_transform(te_band_1)
min_max_scaler = preprocessing.MinMaxScaler()
te_band_2 = min_max_scaler.fit_transform(te_band_2)

temp=np.mean([0 if v == 'na' else v for v in te_ang])
te_ang = [temp if v == 'na' else v for v in te_ang]
min_max_scaler = preprocessing.MinMaxScaler()
te_ang = min_max_scaler.fit_transform(te_ang)
te_ang =te_ang.reshape(-1,1)

Xte = te_band_1
Xte = Xte*np.tile(np.cos(te_ang),(1,5625))
Xte_ = te_band_2
Xte_ = Xte_*np.tile(np.sin(te_ang),(1,5625))
Xte = Xte + Xte_
Xte = Xte.reshape((len(Xte),75,75,1))

Xte.shape
np.save(path+'Xte_dual_ang', Xte)


# Add incidence anggl + Dual Band: way-3
'''
 Xnew= [Band_1*cos(phi); Band_2*sin(phi)], 3D tensor.  Dec.13.2017
'''
min_max_scaler = preprocessing.MinMaxScaler()
te_band_1 = min_max_scaler.fit_transform(te_band_1)
min_max_scaler = preprocessing.MinMaxScaler()
te_band_2 = min_max_scaler.fit_transform(te_band_2)

temp=np.mean([0 if v == 'na' else v for v in te_ang])
te_ang = [temp if v == 'na' else v for v in te_ang]
min_max_scaler = preprocessing.MinMaxScaler()
te_ang = min_max_scaler.fit_transform(te_ang)
te_ang =te_ang.reshape(-1,1)

Xte = te_band_1
Xte = Xte*np.tile(np.cos(te_ang),(1,5625))
Xte_ = te_band_2
Xte_ = Xte_*np.tile(np.sin(te_ang),(1,5625))
Xte = Xte + Xte_
Xte = Xte.reshape((len(Xte),75,75,1))

Xte.shape
np.save(path+'Xte_dual_ang', Xte)