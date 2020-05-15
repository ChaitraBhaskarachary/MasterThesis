from Auxiliar import Healthy as HT
from Auxiliar import AutismTransform as AT
import os
import pandas as pd
import sys

from sklearn.model_selection import train_test_split, cross_val_score

import train_data

# from sklearn.cross_validation import cross_val_score
from sklearn.svm import SVC

import matplotlib.pyplot as plt

plt.style.use("ggplot")

import numpy as np

def plot_trials(X, y, channel_names, title, sfreq=250, trial_len=8):
    X_left = np.mean(X[y == 0,], 0)
    X_right = np.mean(X[y == 1,], 0)

    n_time_samples = sfreq * trial_len
    x = np.linspace(0, trial_len, n_time_samples)
    xf = np.linspace(3.5, trial_len - 1, n_time_samples)
    n_channels = len(channel_names)
    fig, axarr = plt.subplots(n_channels, 2, sharex=True, sharey=True)
    fig.suptitle(title, fontsize=15)

    for ax, left, right, lbl in zip(axarr, X_left, X_right, channel_names):
        ax[0].fill_between(xf, -1, 1, facecolor="yellow", alpha=0.3)
        ax[1].fill_between(xf, -1, 1, facecolor="yellow", alpha=0.3)
        ax[0].set_ylabel(lbl, ha="right", rotation="horizontal")
        ax[0].plot(x, 0)
        ax[1].plot(x, 1)
        ax[0].yaxis.set_major_locator(plt.NullLocator())
        ax[1].yaxis.set_major_locator(plt.NullLocator())
        plt.setp(ax[0].get_yticklabels(), visible=False)
        plt.setp(ax[1].get_yticklabels(), visible=False)

    axarr[0, 0].set_title("Mean of left hand MI trials")
    axarr[0, 1].set_title("Mean of right hand MI trials")

    fig.text(0.06, 0.5, "Channels", ha="center", va="center", rotation="vertical")
    fig.text(0.5, 0.04, "Time (s)", ha="center", va="center", rotation="horizontal")
    z.show(plt, width="800px")
    plt.close()


plot_trials(train_data.X_train, train_data.y_train, train_data.eeg_chans, "Mean of motor imagery trials")