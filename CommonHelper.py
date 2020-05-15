import signal
import numpy as np
import scipy.signal

def get_index_band(rate,lower,upper):
    lower_index=int(lower*rate)
    upper_index=int(upper*rate)
    return[lower_index,upper_index]


def get_power_spectrum(X,channel,fs=250):
    #X=data.X
    #X=data.X[0][0,:]
    #data.X.shape=>(751, 19, 5000)
    #data.X[0][0,:].shape
    # get total subjects
    total_sample_number=X.shape[0]
    # get the datapoints for each signal
    points_per_signal=X.shape[2]
    # create an empty list
    sample_holder=[]
    # loop through trails
    for sample_number in range(0,total_sample_number):
        # create an empty list to hold
        data_channel_holder=[]
        for each_channel in range(0,channel):
            #print("data_channel_holder",data_channel_holder)
            each_signal=X[sample_number,each_channel,:]
            f, Pxx_den = scipy.signal.periodogram(each_signal, fs,scaling="spectrum")
            rate_equi=(points_per_signal/fs)
            #delta power 0-4Hz
            indexs=get_index_band(rate_equi,0,4)
            #delta_power=Pxx_den[indexs[0]:indexs[1]]
            delta_power=scipy.integrate.simps(Pxx_den[indexs[0]:indexs[1]])
            #theta power 4-7hz
            indexs=get_index_band(rate_equi,4,8)
            #print(1,indexs)
            theta_power=scipy.integrate.simps(Pxx_den[indexs[0]:indexs[1]])
            #Alpha power 8-15hz
            indexs=get_index_band(rate_equi,8,12)
            #print(2,indexs)
            alpha_power=scipy.integrate.simps(Pxx_den[indexs[0]:indexs[1]])
            #beta power 16-31hz
            indexs=get_index_band(rate_equi,13,30)
            #print(3,indexs)
            beta_power=scipy.integrate.simps(Pxx_den[indexs[0]:indexs[1]])
            #gamma power 16-31hz
            #indexs=get_index_band(rate_equi,32,32)
            #gamma_power=Pxx_den[indexs[0]:indexs[1]]
            total_power=delta_power+theta_power+alpha_power+beta_power

            data_channel_holder=np.hstack([data_channel_holder,delta_power/total_power,
                                                            theta_power/total_power,
                                                            alpha_power/total_power,
                                                            beta_power/total_power])
            #print(data_channel_holder)
        if(sample_number==0):
            sample_holder=data_channel_holder
        else:
            sample_holder=np.vstack([sample_holder,data_channel_holder])
    return sample_holder
