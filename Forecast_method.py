# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 10:53:16 2018

@author: zhaohuan,ding
"""
import numpy as np
from scipy.fftpack import fft
from scipy import interpolate
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

def ForecastbyFFT(calculate_date,Forecast_time,sample,lowcut,highcut):
    """
    Using FFT to predict time Series

    Parameters
    ----------
    calculate_date : Data used to calculate
.
    Forecast_time : The length, in seconds, of the time series to be predicted
    
    sample : sampling rate 
    
    lowcut,highcut: Upper and lower limits of target frequency band
    
    Returns
    -------
    out : An array of (calculate_num+Forecast_time*sample) in length,
    [calculate_num:calculate_num+Forecast_time*sample] is the predicted value

    """
    t=np.arange(0,100000,1)
    calculate_num=int(len(calculate_date))
    x1 = np.linspace(1,sample,calculate_num)
    x_new = np.linspace(1, sample, sample)

    tck = interpolate.splrep(x1, calculate_date)
    f_temp = interpolate.splev(x_new, tck)
    
    f=fft(f_temp)/len(f_temp)
    ff_temp=abs(f)
    ff=ff_temp[range(int(len(ff_temp)/2))]
    
    ff_theat=ff[lowcut:highcut+1]
    forecast=0*np.sin(2*np.pi*t/sample)
    n=lowcut
    for i in ff_theat:
        forecast+=i*np.sin(2*np.pi*n*t/sample+np.angle(f[n])+np.pi/2)
        n=n+1
  
    forecast=forecast[0:int(calculate_num+Forecast_time*sample)] 
    return forecast



def ForecastbyARmodel(calculate_date,Forecast_time,sample,pbest):    
    """
    Using ARmodel to predict time Series

    Parameters
    ----------
    calculate_date : Data used to calculate
.
    Forecast_time : The length, in seconds, of the time series to be predicted
    
    sample : sampling rate 
    
    pbest: ARmodel order
    
    Returns
    -------
    out : An array of (calculate_num+Forecast_time*sample) in length,
    [calculate_num:calculate_num+Forecast_time*sample] is the predicted value

    """      
    size=int(calculate_date.size/10)
    date=np.zeros(size)
    
    for i in range(0,size):
         date[i]=(calculate_date[i*10+5]+calculate_date[i*10+6])/2
        
    date = date.astype('float32')
    
    matrix_x=np.zeros((date.size-pbest,pbest)) 
    matrix_x=np.matrix(matrix_x)  
    array=date.reshape(date.size) 
    j=0  
    for i in range(0,date.size-pbest):  
        matrix_x[i,0:pbest]=array[j:j+pbest]           
        j=j+1;  
        
    matrix_y=np.array(array[pbest:date.size])  
    matrix_y=matrix_y.reshape(date.size-pbest,1)  
    matrix_y=np.matrix(matrix_y)  
 
    
    fi=np.dot(np.dot((np.dot(matrix_x.T,matrix_x)).I,matrix_x.T),matrix_y)  #A=(XT*X)-1*XT*Y
    
    forecast=np.zeros(calculate_date.size+int(Forecast_time*sample))    
    forecast_temp=np.zeros(date.size+int(Forecast_time*sample/10))
    forecast_temp[0:date.size]=date
    
    for i in range(date.size,date.size+int(Forecast_time*sample/10)): 
        matrix_tempx=np.matrix(forecast_temp[i-pbest:i])
        matrix_tempy=np.dot(matrix_tempx,fi) 
        forecast_temp[i]=matrix_tempy[0][0];       
        
    x1 = np.linspace(1, calculate_date.size+int(Forecast_time*sample),size+int(Forecast_time*sample/10))
    x_new = np.linspace(1, calculate_date.size+int(Forecast_time*sample), calculate_date.size+int(Forecast_time*sample))
    
    tck = interpolate.splrep(x1, forecast_temp)
    forecast = interpolate.splev(x_new, tck)
    forecast[0:calculate_date.size]= calculate_date
    return forecast


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(0,len(dataset)-look_back):
        a = dataset[i:(i+look_back),0]
        dataX.append(a)
        dataY.append(dataset[i + look_back,0])
    return np.array(dataX), np.array(dataY)

def create_datasetX(dataset, look_back=1):
    dataX = []
    for i in range(0,len(dataset)-look_back):
        a = dataset[i+1:(i+look_back+1),0]
        dataX.append(a)
    return np.array(dataX)

def ForecastbyLSTM(calculate_date,Forecast_time,sample,look_back):
    """
    Using LSTM to predict time Series

    Parameters
    ----------
    calculate_date : Data used to calculate
.
    Forecast_time : The length, in seconds, of the time series to be predicted
    
    sample : sampling rate 
    
    lookback: Number of points used for prediction
    
    Returns
    -------
    out : An array of (calculate_num+Forecast_time*sample) in length,
    [calculate_num:calculate_num+Forecast_time*sample] is the predicted value

    """
    size=int(calculate_date.size/10)#æå?00ä¸ªç¹ä¸è¦
    date=np.zeros(size)
    
    for i in range(0,size):
        date[i]=(calculate_date[i*10+5]+calculate_date[i*10+6])/2

    # normalize the dataset    
    date = date.astype('float32')

    scaler = MinMaxScaler(feature_range=(0, 1))
    train  = scaler.fit_transform(date.reshape(-1,1))
    
    trainX, trainY = create_dataset(train, look_back)   
    
    trainX = np.reshape(trainX, (trainX.shape[0],1, look_back))
   
    #å»ºç« LSTM æ¨¡åï¼?
    np.random.seed(7)#è®©éæºæ°ç§å­åºå®
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=0)#è¿­ä»£100æ¬¡ï¼batch sizeä¸?
    
    forecast_temp=np.zeros(date.size+int(Forecast_time*sample/10))
 
    
    for i in range(date.size,date.size+int(Forecast_time*sample/10)): 
        
         trainX=create_datasetX(train, look_back) 
         trainX_temp = np.reshape(trainX, (trainX.shape[0],1, look_back))
         trainPredict = model.predict(trainX_temp)      
         
         
         forecast_temp[i]=trainPredict[len(trainPredict)-1,0]
         train[0:len(train)-1,0]=train[1:len(train),0]
         train[len(train)-1,0]=forecast_temp[i]

        
    forecast_temp = scaler.inverse_transform([forecast_temp])
    forecast_temp[0,0:date.size]=date
    
    x1 = np.linspace(1, calculate_date.size+int(Forecast_time*sample),size+int(Forecast_time*sample/10))
    x_new = np.linspace(1, calculate_date.size+int(Forecast_time*sample), calculate_date.size+int(Forecast_time*sample))    
    tck = interpolate.splrep(x1, forecast_temp[0])
    forecast = interpolate.splev(x_new, tck)

    return forecast