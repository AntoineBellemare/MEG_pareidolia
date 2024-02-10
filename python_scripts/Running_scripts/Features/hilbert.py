import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy.io as sio
import sys
from scipy.signal import hilbert
import h5py

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
# RUN_LIST = {"pareidolia": ["4"], "RS": ["1", "2"]}
# RUN_LIST = {"pareidolia": ["7"], "RS": ["1", "2"]}
SUBJ_LIST = ["11"]
SUBJ_LIST = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]
# SUBJ_LIST = ["04", "07", "08", "09", "10"]
TASK_LIST = ["pareidolia"]

# import pdb; pdb.set_trace


def apply_hilbert(epochs, freq_bands):
    hilbert_results = []
    for min_freq, max_freq in freq_bands:
        band_epochs = epochs.copy().filter(l_freq=min_freq, h_freq=max_freq)
        data = band_epochs.get_data()
        analytic_signal = hilbert(data)
        amplitude_envelope = np.abs(analytic_signal)
        phase_information = np.angle(analytic_signal)
        hilbert_results.append((amplitude_envelope, phase_information))
    return hilbert_results


tmin = -1.5
tmax = 1.5

for subj in SUBJ_LIST:
    for task in TASK_LIST:
        for run in RUN_LIST[task]:
            try:
                # Load epochs
                epochs_name, epochs_path = get_pareidolia_bids(
                    FOLDERPATH, subj, task, run, stage="epo_RT_wide", cond=None
                )
                epochs = mne.read_epochs(epochs_path)
                epochs.pick_types(meg=True, ref_meg=False)
                # Crop epochs
                epochs = epochs.crop(tmin, tmax)
                # Apply Hilbert transform
                hilbert_data = apply_hilbert(epochs, FREQ_BANDS2)

                # Converting complex128 to complex64 for data size reduction
                hilbert_data = np.array(hilbert_data, dtype=np.complex64)

                # Save the Hilbert transform data with compression
                hilbert_file, hilbert_path = get_pareidolia_bids(
                    FOLDERPATH, subj, task, run, stage="Hilbert_RT_wide"
                )
                with h5py.File(hilbert_path, "w") as f:
                    f.create_dataset(
                        "hilbert_data", data=hilbert_data, compression="gzip"
                    )

            except FileNotFoundError as e:
                print(f"File not found for subject {subj}, task {task}, run {run}: {e}")
