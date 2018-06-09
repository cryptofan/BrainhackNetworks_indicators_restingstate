import pandas as pd
import numpy as np
import scipy
import math
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
from pandas import datetime
import math
from sklearn import preprocessing
import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt

import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
#import pandas_datareader.data as web
from scipy import stats



# define the indicators:
def moving_average(x,N): 
    """ return moving average from last N samples: """ 
    SMA = 0
    for i in range(N):
        SMA = SMA + (1/N)*x[-i-1]
    return SMA

def exponential_weighted_moving_average(x,N,alpha): 
    """ return exponential weighted moving average from last N samples: """ 
    EWMA = 0
    s = [x[-1]]
    for i in range(N):
        if i == 0:
            s_i = x[-1]
        else:
            s_i = alpha*x[-i-1] + (1 - alpha)*s[i-1]
        EWMA = EWMA + (1/N)*s_i
        s.append(s_i)
    return EWMA

def relative_strength_index(x,N,alpha):
    """ return relative strength index from the last N samples: """
    ups = np.zeros((N,1))
    downs = np.zeros((N,1))
    for ind in range(N):
        if x[-1-ind] > x[-1-ind-1]:
            ups[ind,0] = x[-1-ind] - x[-1-ind-1]
        if x[-1-ind] < x[-1-ind-1]:
            downs[ind,0] = x[-1-ind-1] - x[-1-ind]
    
    EWMA_ups = 0
    s = [ups[-1]]
    for i in range(N):
        if i == 0:
            s_i = s[-1]
        else:
            s_i = alpha*ups[-i-1] + (1 - alpha)*s[i-1]
        EWMA_ups = EWMA_ups + (1/N)*s_i
        s.append(s_i)
        
    EWMA_downs = 0
    s = [downs[-1]]
    for i in range(N):
        if i == 0:
            s_i = downs[-1]
        else:
            s_i = alpha*downs[-i-1] + (1 - alpha)*s[i-1]
        EWMA_downs = EWMA_downs + (1/N)*s_i
        s.append(s_i)
    
    RS = EWMA_ups/EWMA_downs
    RSI = 100 - (100/(1 + RS))
    return RSI

def Bollinger_bands(x,N,alpha,K):
    """ return Bollinger bands: """ 
    # SMA = 0
    # for i in range(N):
    #    SMA = SMA + (1/N)*x[-i-1]
    SMA = np.mean(x[-N:-1])   
    upper_band = SMA + K*np.std(x[-N:-1])   
    lower_band = SMA - K*np.std(x[-N:-1])   
    return SMA, upper_band, lower_band


base_path = '/scratch/home/g.koehler/BrainhackNetworks_indicators_restingstate/'

# load the data:
df     = pd.read_csv(base_path+'datasets/fMRI/day1/100307.csv',header=None)
data   = df.values
Nvars  = data.shape[1]
labels_Glasser  = pd.read_csv(base_path+'datasets/fMRI/labels_Glasser.csv',header=None)[0].tolist()
df.columns = labels_Glasser
TR     = 0.72 #[s]


# select one time series:
ROI_number = 0
x = data[:,ROI_number]
ROI_name = labels_Glasser[ROI_number]
N = len(x)


# calculate the indicators:
# simple moving average:
N_SMA = 14
SMAvec = np.zeros((N - N_SMA,1))
for ind in range(len(SMAvec)):
    SMAvec[ind] = moving_average(x[:-1-ind],N_SMA)
    
# exponential weighted moving average:    
N_EWMA = 14
alpha = 0.2
EWMAvec = np.zeros((N - N_EWMA,1))
for ind in range(len(EWMAvec)):
    EWMAvec[ind] = exponential_weighted_moving_average(x[:-1-ind],N_SMA,alpha)

# MACD:
alpha = 0.2
N_EWMA1 = 12
EWMAvec1 = np.zeros((N - N_EWMA1,1))
for ind in range(len(EWMAvec1)):
    EWMAvec1[ind] = exponential_weighted_moving_average(x[:-1-ind],N_EWMA1,alpha)
N_EWMA2 = 26
EWMAvec2 = np.zeros((N - N_EWMA2,1))
for ind in range(len(EWMAvec2)):
    EWMAvec2[ind] = exponential_weighted_moving_average(x[:-1-ind],N_EWMA2,alpha)
    
# RSI:
alpha = 0.2
N_RSI = 14
RSIvec = np.zeros((N - N_RSI - 1,1))
for ind in range(len(RSIvec)):
    RSIvec[ind] = relative_strength_index(x[:-1-ind],N_RSI,alpha)
    
# Bollinger bands:
K = 1.0
alpha = 0.2
N_BOLL = 14
BOLLvec_ma = np.zeros((N - N_BOLL,1))
BOLLvec_upper = np.zeros((N - N_BOLL,1))
BOLLvec_lower = np.zeros((N - N_BOLL,1))
for ind in range(len(BOLLvec_ma)):
    Boll = Bollinger_bands(x[:-1-ind],N_BOLL,alpha,K)
    BOLLvec_ma[ind] = Boll[0]
    BOLLvec_upper[ind] = Boll[1]
    BOLLvec_lower[ind] = Boll[2]


# Data Preperation
seq_len = 22
nb_features = 1#len(df.columns)
data = df.loc[:,['V1','V2']].as_matrix() 
nb_features = data.shape[1]#1#len(df.columns)
sequence_length = seq_len + 1 # index starting from 0
result = []

for index in range(len(data) - sequence_length): # maxmimum date = lastest date - sequence length
    result.append(data[index: index + sequence_length]) # index : index + 22days

result = np.array(result)
row = round(0.9 * result.shape[0]) # 90% split

X_train = result[:int(row),:-1,:] # all data until day m
y_train = result[:int(row),-1,-1] # day m + 1 adjusted close price

X_test = result[int(row):,:-1,:]
y_test = result[int(row):,-1,-1] 

#X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], nb_features))
#X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], nb_features))  

# Model Building
d = 0.2
shape = [nb_features, seq_len, 1] # feature, window, output
neurons = [128, 128, 32, 1]

with tf.device("/gpu:0"):

	model = []
	model = Sequential()
	model.add(LSTM(units=neurons[0], return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2])))
	model.add(Dropout(d))
	model.add(LSTM(neurons[1], return_sequences=False, input_shape=(X_train.shape[1],X_train.shape[2])))
	model.add(Dropout(d))
	model.add(Dense(neurons[2],kernel_initializer="uniform",activation='relu'))        
	model.add(Dense(neurons[3],kernel_initializer="uniform",activation='linear'))
	#model.add(Activation('linear'))

	#model = multi_gpu_model(model, gpus=1)
	model.compile(loss='mse', optimizer='rmsprop')
	#model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
	#model.summary()

	# Model Fitting
	model.fit(X_train, y_train, batch_size=512, epochs=50, validation_split=0.1, verbose=1)

# Results
trainScore = model.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))

testScore = model.evaluate(X_test, y_test, verbose=0)
print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))


from scipy import stats
from scipy.stats.stats import pearsonr

prediction = model.predict(X_test)
prediction = np.squeeze(prediction)
pearsonr(prediction,y_test)

