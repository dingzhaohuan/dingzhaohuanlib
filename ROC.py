# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 22:34:41 2018

@author: yourdaddy
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 11:20:01 2018

@author: yourdaddy
"""

# LSTM æ¨¡å
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.fftpack import fft
from scipy import interpolate
import scipy.signal as signal
import scipy.io as sio 
import matplotlib.pyplot as plt
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
dataname=r'C:\Users\yourdaddy\Desktop\F3.mat'
F3o=sio.loadmat(dataname)['F3o'] 

dataname=r'C:\Users\yourdaddy\Desktop\EEG_28case.mat'
case=[]
case.append((sio.loadmat(dataname)['case1'])[0,:])
case.append((sio.loadmat(dataname)['case2'])[0,:])
case.append((sio.loadmat(dataname)['case3'])[0,:])
case.append((sio.loadmat(dataname)['case4'])[0,:])
case.append((sio.loadmat(dataname)['case5'])[0,:])
case.append((sio.loadmat(dataname)['case6'])[0,:])
case.append((sio.loadmat(dataname)['case7'])[0,:])
case.append((sio.loadmat(dataname)['case8'])[0,:])
case.append((sio.loadmat(dataname)['case9'])[0,:])
case.append((sio.loadmat(dataname)['case10'])[0,:])
case.append((sio.loadmat(dataname)['case11'])[0,:])
case.append((sio.loadmat(dataname)['case12'])[0,:])
case.append((sio.loadmat(dataname)['case13'])[0,:])
case.append((sio.loadmat(dataname)['case14'])[0,:])
case.append((sio.loadmat(dataname)['case15'])[0,:])
case.append((sio.loadmat(dataname)['case16'])[0,:])
case.append((sio.loadmat(dataname)['case17'])[0,:])
case.append((sio.loadmat(dataname)['case19'])[0,:])
case.append((sio.loadmat(dataname)['case20'])[0,:])
case.append((sio.loadmat(dataname)['case21'])[0,:])
case.append((sio.loadmat(dataname)['case22'])[0,:])
case.append((sio.loadmat(dataname)['case23'])[0,:])
case.append((sio.loadmat(dataname)['case24'])[0,:])
case.append((sio.loadmat(dataname)['case25'])[0,:])
case.append((sio.loadmat(dataname)['case26'])[0,:])
case.append((sio.loadmat(dataname)['case27'])[0,:])
case.append((sio.loadmat(dataname)['case28'])[0,:])

sample=1000#æ¯ç§éæ ·1000ä¸ªç¹
calculate_num=1000#ç¨æ¥è®¡ç®çç¹æ?
Forecast_time=0.3#æ³è¦é¢æµçæ¶é´ï¼åä½ä¸ºç§
F3=F3o[0,:]
t=np.arange(0,100000,1)
lowcut = 8
highcut = 13
order=1
start_time=8000#èµ·å§æ¶é´
phase=0#é¢æµçç¸ä½?0~360
pbest=8#ARmodelæå¤§é¶æ?

look_back = 40#RNN æ¨¡åé¶æ°

#å¯¹åå§ä¿¡å·åé¢æµä¿¡å·åæ¶è¿ä¸ä¸ªFIRå¸¦éæ»¤æ³¢å¨ï¼?å?Hzï¼?
def butter_bandpass(lowcut,highcut,fs,order=2):
    nyq = 0.5 * fs #å¥å¥æ¯ç¹é¢çä¸ºéæ ·é¢ççä¸å?
    low = lowcut / nyq
    high = highcut / nyq
    b,a = signal.butter(order,[low,high],btype = 'band')
    return b,a

def butter_bandpass_filter(data,lowcut,highcut,fs,order=2):
    b ,a = butter_bandpass(lowcut,highcut,fs,order = order)
    y = signal.lfilter(b,a,data)  ##ä½¿ç¨IIRæFIRæ»¤æ³¢å¨æ²¿ä¸ç»´è¿æ»¤æ°æ?bä¸ºåå­ç³»æ°åé?aä¸ºåæ¯ç³»æ°åé?dataä¸ºæ°æ?
    return y  #yä¸ºæ»¤æ³¢å¨è¾åº

def butter_bandpass_filtfilter(data,lowcut,highcut,fs,order=2):
    b ,a = butter_bandpass(lowcut,highcut,fs,order = order)
    y = signal.filtfilt(b, a, data)
    return y

def Preprocess(data):#data为原始信号，dataout为滤波信号，预处�?      
    dataout=butter_bandpass_filtfilter(data, lowcut, highcut, sample, order)    
    return dataout

def create_dataset(dataset, look_back=1):#æè®­ç»æ°æ®æåæè¾å¥æ°æ®åè¾åºæ°æ?
    dataX, dataY = [], []
    for i in range(0,len(dataset)-look_back):
        a = dataset[i:(i+look_back),0]
        dataX.append(a)
        dataY.append(dataset[i + look_back,0])
    return np.array(dataX), np.array(dataY)

def create_datasetX(dataset, look_back=1):#æè®­ç»æ°æ®æåæè¾å¥æ°æ®
    dataX = []
    for i in range(0,len(dataset)-look_back):
        a = dataset[i+1:(i+look_back+1),0]
        dataX.append(a)
    return np.array(dataX)

#åºäºFFTçé¢æµæ¹æ³?
def ForecastbyFFT(calculate_date,sample):
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


#åºäºARmodleçé¢æµæ¹æ³?
def ForecastbyARmodel(calculate_date,sample):    

    size=int(calculate_date.size/10)#ææå?00ä¸ªç¹ä¸è¦ï¼é¿åè¿æ»¤æ³¢å¨æ¶è¾¹ç¼ä½ç½®çå½±å?
    date=np.zeros(size)
    
    for i in range(0,size):
        date[i]=(calculate_date[i*10]+calculate_date[i*10])/2#çº¿æ§æå?
        
    date = date.astype('float32')
    
    matrix_x=np.zeros((date.size-pbest,pbest)) #æå»ºä¸ä¸ªN-p*pçæ°ç»? 
    matrix_x=np.matrix(matrix_x)  
    array=date.reshape(date.size) #å¤å¶ä¸ä¸ªåæ¥çæ°ç»
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

def output(data,phase=0):#data1ä¸ºåå§ä¿¡å·ï¼data2ä¸ºé¢æµä¿¡å·ï¼phaseä¸ºç®æ ç¸ä½?
    da=data[calculate_num:int(calculate_num+Forecast_time*sample)]    
    h=hilbert(da)
    ph =  (np.angle(h)+np.pi/2)*180/np.pi
    t=-1
    for i in np.arange(20,int(Forecast_time*sample),1): 
        if ph[i] > phase and ph[i] < (phase+1):
            t=i
            break
    
    return  t



def hilbert_phase(data1,data2,phase=0):#data1ä¸ºåå§ä¿¡å·ï¼data2ä¸ºé¢æµä¿¡å·ï¼phaseä¸ºç®æ ç¸ä½?
    da1=data1[0:int(calculate_num+Forecast_time*sample)]
    da2=data2[0:int(calculate_num+Forecast_time*sample)]    
    
    h1=hilbert(da1)
    h2=hilbert(da2)
    ph2 = (np.angle(h2)+np.pi/2)*180/np.pi
    t=-1
    for i in np.arange(calculate_num+20,int(calculate_num++Forecast_time*sample),1): 
        if ph2[i] > phase and ph2[i] < (phase+5):
            t=i
            break        
            
    
    ph1_ = ((np.angle(h1)+np.pi/2+2*np.pi)*180/np.pi)%360#np.unwrap(np.angle(h1)+np.pi/2)#unwrapå·åºå½æ°ï¼è®©ç¸è§è¿ç»­
    ph2_ = ((np.angle(h2)+np.pi/2+2*np.pi)*180/np.pi)%360#np.unwrap(np.angle(h2)+np.pi/2)
    er_= (np.unwrap(ph2_-ph1_,discont=90)+900)%360-180#è®©ç»æè½å?180~180åº¦ä¹é´ï¼ä¸æ è·³å
#    er= np.unwrap(ph2-ph1,discont=90)#è®©ç»æè½å?180~180åº¦ä¹é´ï¼ä¸æ è·³å
    er=er_[calculate_num:int(Forecast_time*sample)+calculate_num]
    ph1=ph1_[calculate_num:int(Forecast_time*sample)+calculate_num]
    ph2=ph2_[calculate_num:int(Forecast_time*sample)+calculate_num]
    return  ph1, ph2, er,t


def Phase_ROC(data,phase_threshold):
    er_list_A=[]
    tp=0#实际与预测值都为真
    fn=0#实际为真，预测值为�?
    fp=0#实际为否，预测值为�?
    tn=0#实际与预测值都为否
    er_p=0
    er_t=0
    
    for j in range(0,27):
        for i in range(0,40):
            data1=data[j]
            epoch=data1[i*1000:int(calculate_num+Forecast_time*sample)+i*1000]                       
#            f_temp=epoch[0:calculate_num]#用来计算的数�?
#            ff_temp=ForecastbyARmodel(f_temp,sample)#预测
#            ff_temp=ForecastbyFFT(f_temp,sample)
#            ff_forecast=Preprocess(ff_temp)
#            epoch_filt=Preprocess(epoch)#1.实际脑电预处�?  


            epoch_filt=Preprocess(epoch)#1.ä¿¡å·é¢å¤ç?
            f_temp=epoch_filt[0:calculate_num]
#            ff_forecast=ForecastbyARmodel(f_temp,sample)
            ff_forecast=ForecastbyFFT(f_temp,sample)


                     
            
            ph1, ph2,er,t=hilbert_phase(epoch_filt,ff_forecast,phase)                      
            er_list_A.append(er[10])            
            er_t=((ph1[10]-phase)+900)%360-180
            er_p=((ph2[10]-phase)+900)%360-180
            
            if (er_t<=30 and er_t>-30):
                 if (er_p<=phase_threshold and er_p>-phase_threshold):    
                     tp=tp+1
                 else:
                     fn=fn+1
            else:
                 if (er_p<=phase_threshold and er_p>-phase_threshold):
                     fp=fp+1
                 else:
                     tn=tn+1       

    fpr=fp/(fp+tn)        
    tpr=tp/(tp+fn)            
    return  fpr, tpr 

def SNR(data):
    snr=[]   
    snr_new=[]
    for j in range(0,27):
        data1=data[j]
        data2=Preprocess(data1)
        
        f=fft(data1)/len(data1)
        ff_temp=abs(f)
        ff1=ff_temp[range(int(len(ff_temp)/2))]   

        f=fft(data2)/len(data2)
        ff_temp=abs(f)
        ff2=ff_temp[range(int(len(ff_temp)/2))]   
        
        
        inte=int(len(ff_temp)/sample)
        
        fs=0
        fn=0
        for i in range(0,int(len(ff_temp)/2)):
            fn=fn+ff1[i]
        for i in range(inte,inte*30):
            fs=fs+ff1[i]    
        snr.append(fs/fn)

        fs=0
        fn=0
        for i in range(0,int(len(ff_temp)/2)):
            fn=fn+ff2[i]
        for i in range(inte,inte*30):
            fs=fs+ff2[i]    
        snr_new.append(fs/fn)
        
    return  snr,snr_new



def _ROC(data):
    fpr_list=[]
    tpr_list=[]
    for j in np.arange(0,190,10):
        fpr, tpr=Phase_ROC(data,j) 
        fpr_list.append(fpr)
        tpr_list.append(tpr)
    return  fpr_list, tpr_list


def _AUC(data):
    _auc=[] 
    global pbest
    for i in range(0,1):
        fpr_list, tpr_list=_ROC(data)      
        _auc.append(auc(fpr_list, tpr_list))
        pbest=pbest+1
    return _auc,fpr_list, tpr_list
 
#æ ¸å¿ç®æ³
ctt=case[8]
start = float(time.time())#è®°å½æ¶é´

#epoch=ctt[start_time:int(calculate_num+Forecast_time*sample)+start_time]
#f_temp=epoch[0:calculate_num]#用来计算的数�?
#ff_temp=ForecastbyARmodel(f_temp,sample)#预测
#ff_temp=ForecastbyFFT(f_temp,sample)
#ff_forecast=Preprocess(ff_temp)


epoch=ctt[start_time:int(calculate_num+Forecast_time*sample)+start_time]
epoch_filt=Preprocess(epoch)#1.ä¿¡å·é¢å¤ç?
f_temp=epoch_filt[0:calculate_num]
ff_forecast=ForecastbyARmodel(f_temp,sample)




end = float(time.time())#è®°å½æ¶é´
print(end - start)#è®°å½æ¶é´

epoch_filt=Preprocess(epoch)#1.实际脑电预处�?
ph1, ph2,er,t=hilbert_phase(epoch_filt,ff_forecast,phase)


#fpr_list, tpr_list=_ROC(case)
#arr_auc,fpr_list,tpr_list=_AUC(case)
#snr,snr_new=SNR(case)
bb=F3[0:5000]
cc=Preprocess(bb)


#ç»å¶ç»æ
plt.figure(1)#预处理前后，信号对比
plt.plot(np.arange(-calculate_num,int(Forecast_time*sample),1),epoch,'-b',label='real')
plt.plot(np.arange(-calculate_num,int(Forecast_time*sample),1),epoch_filt,'-y',label='real_flit')
new_ticks = np.linspace(-calculate_num, int(Forecast_time*sample), (calculate_num/sample+Forecast_time)*10+1)
plt.xticks(new_ticks)
plt.xlim(-calculate_num,int(Forecast_time*sample))
plt.xlabel('Time(mS)')
plt.ylabel('μV')
plt.legend(loc='lower left')

plt.figure(2)#é¢æµæ³¢å½¢åå®éæ³¢å½¢å¯¹æ¯?
#plt.plot(np.arange(-calculate_num,int(Forecast_time*sample),1),ff_temp[0:calculate_num+int(Forecast_time*sample)],'--k',label='forecast')
plt.plot(np.arange(-calculate_num,int(Forecast_time*sample),1),epoch+10,'-y',label='real_flit')
plt.plot([0,0],[-25,25],'r-')

new_ticks = np.linspace(-calculate_num, int(Forecast_time*sample), (calculate_num/sample+Forecast_time)*10+1)
plt.xticks(new_ticks)
plt.xlim(-calculate_num,int(Forecast_time*sample))
plt.ylim(-80,80)


plt.legend(loc='lower left')

plt.figure(3)#é¢æµæ³¢å½¢åå®éæ³¢å½¢å¯¹æ¯?
plt.plot(np.arange(-calculate_num,int(Forecast_time*sample),1),ff_forecast[0:calculate_num+int(Forecast_time*sample)],'--k',label='forecast')
plt.plot(np.arange(-calculate_num,int(Forecast_time*sample),1),epoch_filt,'-k',label='real_flit')
plt.plot([0,0],[-25,25],'r-')

new_ticks = np.linspace(-calculate_num, int(Forecast_time*sample), (calculate_num/sample+Forecast_time)*10+1)
plt.xticks(new_ticks)
plt.xlim(-calculate_num,int(Forecast_time*sample))
plt.ylim(-80,80)

plt.xlabel('Time(mS)')
plt.ylabel('Î¼V')
plt.legend(loc='lower left')


plt.figure(4)#é¢æµæ³¢å½¢ç¸ä½åå®éç¸ä½åå·?
plt.plot(np.arange(0,int(Forecast_time*sample),1),er,'-b')
#if t!=-1:
 #   plt.plot([t,t],[er[t]-10,er[t]+10],'r-',label='trigger error')
plt.xlim(0,int(Forecast_time*sample))
new_ticks = np.linspace(0, int(Forecast_time*sample), Forecast_time*50+1)
plt.xticks(new_ticks)
plt.xlabel('Time(mS)')
plt.ylabel('phase')
plt.legend(loc='lower left')





plt.figure(5)
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr_list, tpr_list, color='k',
         lw=lw, label='ROC' ) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='k', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('P=8')
plt.legend(loc="lower right")




