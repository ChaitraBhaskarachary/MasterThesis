import logging

import pandas as pd
import scipy
import scipy.signal

import sys
import os
import mne
import numpy as np

# /home/gari/anaconda3/envs/EEG/lib/python3.7/site-packages/Braindecode-0.4.85-py3.7.egg/braindecode/datautil
# C:/Users/Akshay/PycharmProjects/austismthesis
# Absolute path of .current script
script_pos = os.path.dirname(os.path.abspath(__file__))
# script_pos = os.path.dirname(__file__)
print("script_pos",script_pos)

Auxiliar_pos = script_pos + "/Auxiliar"

# Include Auxiliar_pos in the current python enviroment
if not Auxiliar_pos in sys.path:
    sys.path.append(Auxiliar_pos)

Data_pos = script_pos + "/Data"
# Include Data_pos in the current python enviroment
if not Data_pos in sys.path:
   sys.path.append(Data_pos)
# Picke_pos = script_pos + "/picke_files"

# My custom package for transforming the
# healthy database csv files to
# mne objects
from Auxiliar import Healthy as HBT
from Auxiliar import AutismTransform as AT
from Auxiliar import PairSignalConcat
from Auxiliar import OFHandlers as OFH
from Auxiliar import CommonHelper as CH

path_Autism_data = Data_pos + "/Autism/"
Autism_subjects = os.listdir(path_Autism_data)
print("total Autism_subjects available", len(Autism_subjects))

path_healthy_data = Data_pos + "/Healthy/"
healthy_subjects = os.listdir(path_healthy_data)
print("total healthy_subjects available",len(healthy_subjects))

i = 0
for each_healthy in healthy_subjects[0:80]:
    print("*" * 100)
    #print(i)

    each_healthy = healthy_subjects[i]
    raw_healthy, eeg_chans_he = HBT.hbn_raw(subject=each_healthy, path_absolute='D:/Healthy/')

    # Autism
    each_autism = Autism_subjects[i]
    raw_Autism, eeg_chans = AT.autism_raw(subject=each_autism,path_absolute='D:/Autism/')
    # merge signals
    signal_healthy = PairSignalConcat.concat_prepare_cnn(raw_healthy)
    signal_Autism = PairSignalConcat.concat_prepare_cnn(raw_Autism)
    signal_Autism.y = np.where(signal_Autism.y==0,1,signal_Autism)

    if (i == 0):
        print(i, each_healthy, each_autism)
        concat_signal = signal_healthy
        concat_signal.X = np.vstack([signal_healthy.X, signal_Autism.X])
        concat_signal.y = np.concatenate((signal_healthy.y, signal_Autism.y), axis=0)
        print("concat_signal.X.shape", concat_signal.X.shape)
        print("concat_signal.y.shape", concat_signal.y.shape)
    else:
        print("+" * 100)
        print(i, each_healthy, each_autism)
        concat_signal.X = np.vstack([concat_signal.X, signal_healthy.X, signal_Autism.X])
        concat_signal.y= np.concatenate((concat_signal.y, signal_healthy.y, signal_Autism.y), axis=0)
        print("concat_signal.X.shape", concat_signal.X.shape)
        print("concat_signal.y.shape", concat_signal.y.shape)
    i = i + 1
print("concat_signal.X.shape",concat_signal.X.shape)

OFH.OFHandlers.save_object("train_combined_signal.file",concat_signal)
