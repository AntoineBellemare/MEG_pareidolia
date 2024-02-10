import sys

sys.path.insert(0, "C:/Users/Antoine/github/MEG_pareidolia/python_scripts/Functions")
import MEG_pareidolia_utils
from MEG_pareidolia_utils import *
import mne
from mne.datasets.brainstorm import bst_raw
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy.io as sio

# import brainpipe
# from brainpipe import feature
import pandas as pd
import scipy.io as sio
from scipy.io import savemat, loadmat

# import PARAMS
# from PARAMS import *
from fooof import FOOOF
import argparse

RUN_LIST = {"pareidolia": ["1", "2", "3", "4", "5", "6", "7", "8"]}

SUBJ_LIST = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]
# SUBJ_LIST = ['00']
TASK_LIST = ["pareidolia"]
task = "pareidolia"
max_freq = 100
# import pdb; pdb.set_trace


# This function is used to compute power values for each frequency bands on each epochs
def compute_FOOOF(
    epochs,
    function=psd_welch,
    tmin=None,
    tmax=None,
    max_freq=150,
    participant=None,
    bloc=None,
):
    # epochs are cropped as desire (tmin could be before '0', ex: -1.5, depending on the values used during epoching)
    epochs = epochs.apply_baseline((-2, -1.5))
    epochs.pick_types(meg=True, ref_meg=False)
    epochs = epochs.crop(tmin, tmax)
    print(epochs.get_data().shape)
    epo_data = epochs.get_data()
    exps = []
    df = pd.DataFrame(
        columns=[
            "offset",
            "exp",
            "peaks_center",
            "amps",
            "width",
            "detrend_psd",
            "freq_range",
            "freq_res",
            "participant",
            "bloc",
            "trial",
            "electrode",
            "corr_spec",
            "r2",
        ]
    )
    # This loop iterates for each epoch
    i = 0
    for t in range(len(epo_data)):
        print("TRIAL: ", str(t))
        exps_elec = []
        for elec in range(len(epo_data[t])):
            (
                offset,
                exp,
                cf,
                amp,
                width,
                detrend_psd,
                freq_range,
                freq_res,
                corr_spec,
                r2,
                estimated_slope,
            ) = FOOOF_aperiodic(
                epo_data[t][elec],
                1200,
                precision=1,
                max_freq=max_freq,
                noverlap=None,
                nperseg=None,
                nfft=None,
                extended_returns=False,
                graph=False,
            )
            df.loc[i] = (
                offset,
                exp,
                cf,
                amp,
                width,
                detrend_psd,
                freq_range,
                freq_res,
                participant,
                bloc,
                t,
                elec,
                corr_spec,
                r2,
            )
            i += 1

    return df


for subj in SUBJ_LIST:
    print("SUBJ : ", str(subj))
    for run in RUN_LIST[task]:
        try:
            print("RUN: ", str(run))
            epochs_name, epochs_path = get_pareidolia_bids(
                FOLDERPATH, subj, "pareidolia", run, stage="epo_RT", cond=None
            )
            epochs = mne.read_epochs(epochs_path)
            # Si vous voulez comparer les epochs entières (8sec), il est mieux de laisser de côté le début et la fin des epochs.
            fooof_df = compute_FOOOF(
                epochs,
                function=psd_welch,
                tmin=-1.5,
                tmax=-0.5,
                max_freq=max_freq,
                participant=int(subj),
                bloc=run,
            )
            # le nom du stage doit commencer par PSD, la fin du nom est à votre choix
            FOOOF_file, FOOOF_path = get_pareidolia_bids(
                FOLDERPATH, subj, "pareidolia", run, stage="FOOOF_RT_new"
            )
            fooof_df.to_csv(FOOOF_path)

        except FileNotFoundError:
            pass
