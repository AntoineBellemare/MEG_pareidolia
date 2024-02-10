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
sf = 1200
# import pdb; pdb.set_trace


# This function is used to compute power values for each frequency bands on each epochs
def compute_psd(epoch, FREQ_BANDS, function=psd_array_welch):
    # epochs are cropped as desire (tmin could be before '0', ex: -1.5, depending on the values used during epoching)
    # epochs = epochs.apply_baseline((-1, -0.5))
    PSDs = []
    # This loop iterates for each epoch

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
        PSDs.append(psds_mean)
    PSDs = np.array(PSDs)
    return PSDs


##Compute_PSD
for subj in SUBJ_LIST:
    for task in TASK_LIST:
        for run in RUN_LIST[task]:
            try:
                source_names, source_paths = get_pareidolia_bids(
                    FOLDERPATH, subj, task, run, stage="stc", cond=None
                )
                # Si vous voulez comparer les epochs entières (8sec), il est mieux de laisser de côté le début et la fin des epochs.
                # loop through the source files which are in the same order as the epochs
                all_psds = []
                for i, source in enumerate(source_paths):
                    # load the source file
                    stc = mne.read_source_estimate(source)
                    # select the time period of interest
                    # extract the data
                    data = stc.data
                    # keep the last 8 seconds of the data
                    data = data[:, -9600:]
                    # compute the PSDs
                    psds_welch, freqs = psd_array_welch(data, sfreq=1200, n_jobs=-1)
                    all_psds.append(psds_welch)

                all_psds = np.array(all_psds)
                # save the PSDs
                psds_file, psds_path = get_pareidolia_bids(
                    FOLDERPATH, subj, task, run, stage="PSD_source"
                )
                savemat(psds_path, {"PSD": psds_welch})
            except (FileNotFoundError, OSError):
                pass
