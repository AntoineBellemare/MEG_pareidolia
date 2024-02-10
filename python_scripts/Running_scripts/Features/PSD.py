import mne
from mne.datasets.brainstorm import bst_raw
from mne.time_frequency import psd_array_welch, psd_array_multitaper
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy.io as sio
import sys

sys.path.insert(0, "C:/Users/Antoine/github/MEG_pareidolia/python_scripts/Functions")
import MEG_pareidolia_utils

# import brainpipe
# from brainpipe import feature
import pandas as pd
import scipy.io as sio
from scipy.io import savemat, loadmat
from MEG_pareidolia_utils import *
import PARAMS
from PARAMS import *

RUN_LIST = {"pareidolia": ["1", "2", "3", "4", "5", "6", "7", "8"], "RS": ["1", "2"]}
# RUN_LIST = {"pareidolia": ["7"], "RS": ["1", "2"]}
SUBJ_LIST = ["08", "09", "10", "11"]
SUBJ_LIST = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]
TASK_LIST = ["pareidolia"]

# import pdb; pdb.set_trace


# This function is used to compute power values for each frequency bands on each epochs
def compute_psd(epochs, FREQ_BANDS, function=psd_array_welch, tmin=None, tmax=None):
    # epochs are cropped as desire (tmin could be before '0', ex: -1.5, depending on the values used during epoching)
    # epochs = epochs.apply_baseline((-1, -0.5))
    epochs.pick_types(meg=True, ref_meg=False)
    epochs = epochs.crop(tmin, tmax)
    print(epochs.get_data().shape)
    PSDs = []
    # This loop iterates for each epoch
    for t in range(len(epochs[:].get_data())):
        psds_temp = []
        picks = mne.pick_types(
            epochs.info, meg=True, ref_meg=False, eeg=False, eog=False, stim=False
        )
        for min_, max_ in FREQ_BANDS:
            epochs_data = epochs[t].get_data()
            # select MEG channels
            psds, freqs = function(
                epochs_data, sfreq=1200, fmin=min_, fmax=max_, n_jobs=-1
            )  # PSDs are calculated with this function, giving power values and corresponding frequency bins as output
            psds = 10.0 * np.log10(
                psds
            )  # PSDs values are transformed in log scale to compensate for the 1/f natural slope
            psds_mean = np.average(psds, axis=0)  # Get rid of an empty dimension
            psds_mean = np.average(
                psds_mean, axis=1
            )  # Average across bins to obtain one value for the entire frequency range
            psds_temp.append(psds_mean)
        PSDs.append(psds_temp)
    PSDs = np.array(PSDs)
    return PSDs


##Compute_PSD
for subj in SUBJ_LIST:
    for task in TASK_LIST:
        for run in RUN_LIST[task]:
            try:
                epochs_name, epochs_path = get_pareidolia_bids(
                    FOLDERPATH, subj, task, run, stage="epo_RT", cond=None
                )
                epochs = mne.read_epochs(epochs_path)
                # Si vous voulez comparer les epochs entières (8sec), il est mieux de laisser de côté le début et la fin des epochs.
                psds_welch = compute_psd(
                    epochs,
                    FREQ_BANDS_multigamma,
                    psd_array_multitaper,
                    tmin=-1.5,
                    tmax=-0.5,
                )
                # le nom du stage doit commencer par PSD, la fin du nom est à votre choix
                psds_file, psds_path = get_pareidolia_bids(
                    FOLDERPATH, subj, task, run, stage="PSD_RT_large_gamma"
                )
                savemat(psds_path, {"PSD": psds_welch})
            except FileNotFoundError:
                pass
