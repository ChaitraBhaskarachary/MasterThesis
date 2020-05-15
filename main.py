from Auxiliar import OFHandlers as OFH
import time
import numpy as np
import pandas as pd
import scipy.signal
from Auxiliar import CommonHelper as CH
import scipy

train = OFH.OFHandlers.load_object('C:/Users/Akshay/PycharmProjects/austismthesis/concat_signal_X.file')
test = OFH.OFHandlers.load_object('C:/Users/Akshay/PycharmProjects/austismthesis/concat_signal_test.file')


band_features=CH.get_power_spectrum(train.X,110)
df=pd.DataFrame(band_features)
df_y=pd.DataFrame(train.y,columns=["target"])
prepared_data_set=pd.concat([df,df_y], axis=1)
print(prepared_data_set)
OFH.OFHandlers.save_object("preprocessed_train_data.file",prepared_data_set)

# feature extraction for test data
band_features=CH.get_power_spectrum(test.X,110)
df=pd.DataFrame(band_features)
df_y=pd.DataFrame(test.y,columns=["target"])
prepared_data_set=pd.concat([df,df_y], axis=1)
OFH.OFHandlers.save_object("preprocessed_test_data.file",prepared_data_set)

