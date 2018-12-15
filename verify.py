# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 17:10:44 2018

@author: zhaohuan,ding
"""
from scipy.signal import hilbert
import numpy as np
from scipy.fftpack import fft

def hilbert_phase(data1,data2):
    """
    calculate two time series instant phases and their differences

    Parameters
    ----------
    data1,data2 : Data used to calculate.
    
    Returns
    -------
    ph1, ph2, er : time series instant phases and their differences

    """
    
    h1=hilbert(data1)
    h2=hilbert(data2)
    ph2 = (np.angle(h2)+np.pi/2)*180/np.pi
    
    ph1 = ((np.angle(h1)+np.pi/2+2*np.pi)*180/np.pi)%360
    ph2 = ((np.angle(h2)+np.pi/2+2*np.pi)*180/np.pi)%360
    er= (np.unwrap(ph2-ph1,discont=90)+900)%360-180

    return  ph1, ph2, er


def SNR(data,sample):

    data1=data
    
    f=fft(data1)/len(data1)
    ff_temp=abs(f)
    ff1=ff_temp[range(int(len(ff_temp)/2))]   
    inte=int(len(ff_temp)/sample)
    
    fs=0
    fn=0
    for i in range(0,int(len(ff_temp)/2)):
        fn=fn+ff1[i]
    for i in range(inte,inte*30):
        fs=fs+ff1[i]    
    snr=fs/fn
        
    return  snr