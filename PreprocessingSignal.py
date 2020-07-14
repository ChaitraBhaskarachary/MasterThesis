from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import (
    bandpass_cnt,
    exponential_running_standardize,
)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne

import mne
from collections import OrderedDict
import numpy as np


def concat_prepare_cnn(input_signal):
    # In this particular data set
    # it was required by the author of it,
    # that for preventing the algorithm
    # to pick on data of the eye movement,
    # a high band filter of Hz had to
    # be implimented.
    low_cut_hz = 1

    # The authors prove both configuration >38  an 38< frequenzy
    # in the current experiment, we see that band pass filter will take
    # Theta to part of Gamma frequenzy band
    # whihch what Filter Bank Commun spatial patters would do.
    # This value is a hiperpartemer that should be ajusted
    # per data set... In my opinion.
    high_cut_hz = 40

    # factor for exponential smothing
    # are this numbers usually setup used
    # on neuro sciencie?
    factor_new = 1e-3

    # initianlization values for the the mean and variance,
    # see prior discussion
    init_block_size = 1000

    # model = "shallow"  #'shallow' or 'deep'
    # GPU support
    # cuda = True

    # It was stated in the paper [1] that
    # "trial   window   for   later   experiments   with   convolutional
    # networks,  that  is,  from  0.5  to  4  s."
    # 0- 20s?
    # so "ival" variable simple states what milisecond interval to analize
    # per trial.
    ival = [0, 20000]

    # An epoch increase every time the whole training data point
    # had been input to the network. An epoch is not a batch
    # example, if we have 100 training data points
    # and we use batch_size 10, it will take 10 iterations of
    # batch_size to reach 1 epoch.
    # max_epochs = 1600

    # max_increase_epochs = 160

    # 60 data point per forward-backwards propagation
    # batch_size = 60

    # pertecentage of data to be used as test-set
    valid_set_fraction = 0.2

    gdf_events = mne.find_events(input_signal)

    input_signal = input_signal.drop_channels(["stim"])

    raw_training_signal = input_signal.get_data()

    print("data shape:", raw_training_signal.shape)

    for i_chan in range(raw_training_signal.shape[0]):
        # first set to nan, than replace nans by nanmean.
        this_chan = raw_training_signal[i_chan]
        raw_training_signal[i_chan] = np.where(
            this_chan == np.min(this_chan), np.nan, this_chan
        )
        mask = np.isnan(raw_training_signal[i_chan])
        chan_mean = np.nanmean(raw_training_signal[i_chan])
        raw_training_signal[i_chan, mask] = chan_mean

    # Reconstruct
    input_signal = mne.io.RawArray(raw_training_signal, input_signal.info, verbose="WARNING")
    # append the extracted events
    # raw_gdf_training_signal
    # raw_gdf_training_signal
    input_signal.info["events"] = gdf_events

    train_cnt = input_signal
    # lets convert to millvolt for numerical stability of next operations
    train_cnt = mne_apply(lambda a: a * 1e6, train_cnt)
    train_cnt = mne_apply(
        lambda a: bandpass_cnt(
            a,
            low_cut_hz,
            high_cut_hz,
            train_cnt.info["sfreq"],
            filt_order=3,
            axis=1,
        ),
        train_cnt,
    )

    train_cnt = mne_apply(
        lambda a: exponential_running_standardize(
            a.T,
            factor_new=factor_new,
            init_block_size=init_block_size,
            eps=1e-4,
        ).T,
        train_cnt,
    )

    marker_def = OrderedDict(
        [
            ("ec", [30])
        ]
    )

    train_set = create_signal_target_from_raw_mne(train_cnt, marker_def, ival)
    return train_set
