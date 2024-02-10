import mne
from mne.datasets.brainstorm import bst_raw
from mne.time_frequency import psd_array_welch, psd_array_multitaper
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy.io as sio
import sys
import numpy

sys.path.insert(0, "C:/Users/Antoine/github/MEG_pareidolia/python_scripts/Functions")
import MEG_pareidolia_utils

import PARAMS
from PARAMS import *
import fooof
from fooof import FOOOF

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
TASK_LIST = ["RS"]

# import pdb; pdb.set_trace
SFREQ = 1200  # Sampling frequency
# FREQ_RESOLUTION = 0.1  # Desired frequency resolution
NPERSEG = int(SFREQ / 4)  # Number of points per segment
NPERSEG_short = int(SFREQ / 4)
NFFT = int(SFREQ // 1)  # Number of points in FFT
BANDWIDTH = 8  # Bandwidth of windows in multitaper analysis


def compute_FOOOF(
    freqs1,
    psd,
    sf,
    precision=0.1,
    max_freq=80,
    min_freq=2,
    noverlap=None,
    nperseg=None,
    nfft=None,
    extended_returns=False,
    graph=False,
):
    if nperseg is None:
        mult = 1 / precision
        nfft = sf * mult
        nperseg = nfft
        noverlap = nperseg // 10
    fm = FOOOF(peak_width_limits=[1, 7], max_n_peaks=8, min_peak_height=0.3)

    freq_range = [min_freq, max_freq]
    fm.fit(freqs1, psd, freq_range)
    if graph is True:
        fm.report(freqs1, psd)

    try:
        offset = fm.get_params("aperiodic_params")[0]
        exp = fm.get_params("aperiodic_params")[1]
        print("OFFSET: ", offset)
        print("EXP: ", exp)
    except:
        offset = "NaN"
        exp = "NaN"
    try:
        cf = [x[0] for x in fm.get_params("peak_params")]
        amp = [x[1] for x in fm.get_params("peak_params")]
        width = [x[2] for x in fm.get_params("peak_params")]
    except IndexError:
        cf = "NaN"
        amp = "NaN"
        width = "NaN"
    corrected_spectrum = fm.power_spectrum - fm._ap_fit
    r2 = fm.r_squared_
    print("R2: ", r2)
    return (
        offset,
        exp,
        cf,
        amp,
        width,
        fm.fooofed_spectrum_,
        fm.freq_range,
        fm.freq_res,
        corrected_spectrum,
        r2,
        fm._ap_fit,
    )


# This function is used to compute power values for each frequency bands on each epochs
def compute_psd(
    epochs,
    FREQ_BANDS,
    function=psd_array_welch,
    tmin=None,
    tmax=None,
    fmin=4,
    fmax=60,
    run=None,
):
    # epochs are cropped as desire (tmin could be before '0', ex: -1.5, depending on the values used during epoching)
    # epochs = epochs.apply_baseline((-1, -0.5))
    epochs.pick_types(meg=True, ref_meg=False)
    epochs = epochs.crop(tmin, tmax)
    print(epochs.get_data().shape)

    # condition 1 is when metadata 'parei' = 0, condition 2 is when metadata 'parei' = 1
    PSDs_cond1 = []
    PSDs_cond2 = []

    for t in range(len(epochs[:].get_data())):
        epochs_data = epochs[t].get_data()
        psds, freqs = function(
            epochs_data,
            sfreq=SFREQ,
            fmin=fmin,
            fmax=fmax,
            n_fft=NFFT,
            n_per_seg=NPERSEG,
            n_jobs=-1,
        )
        print("PSDS LONG SHAPE: ", psds.shape)
        print("FREQS LONG SHAPE: ", freqs.shape)
        # print min and max of freqs
        print("FREQS MIN: ", np.min(freqs))
        print("FREQS MAX: ", np.max(freqs))
        if run == "1":
            PSDs_cond1.append(psds)
        elif run == "2":
            PSDs_cond2.append(psds)
    # average across epochs

    PSDs_cond1 = np.array(PSDs_cond1)
    PSDs_cond1 = np.average(PSDs_cond1, axis=0)

    PSDs_cond2 = np.array(PSDs_cond2)
    PSDs_cond2 = np.average(PSDs_cond2, axis=0)
    return PSDs_cond1, PSDs_cond2, freqs


def compute_bands(
    epochs,
    FREQ_BANDS,
    function=psd_array_welch,
    tmin=None,
    tmax=None,
    slope_cond=None,
    fmin=4,
    fmax=60,
    run=None,
):
    # epochs are cropped as desire (tmin could be before '0', ex: -1.5, depending on the values used during epoching)
    # epochs = epochs.apply_baseline((-1, -0.5))
    epochs.pick_types(meg=True, ref_meg=False)
    epochs = epochs.crop(tmin, tmax)
    print(epochs.get_data().shape)
    PSDs = []
    # This loop iterates for each epoch
    for t in range(len(epochs[:].get_data())):
        psds_temp = []
        epochs_data = epochs[t].get_data()
        # select MEG channels
        psds, freqs = function(
            epochs_data,
            sfreq=1200,
            fmin=fmin,
            fmax=fmax,
            n_jobs=-1,
            n_fft=NFFT,
            n_per_seg=NPERSEG_short,
        )  # PSDs are calculated with this function, giving power values and corresponding frequency bins as output
        psds = 10.0 * np.log10(psds)
        # get rid of an empty dimension
        psds = np.average(psds, axis=0)
        print("PSDS SHORT SHAPE: ", psds.shape)
        for elec in range(len(psds)):
            psds[elec] = psds[elec] - slope_cond[elec]

        # create mask for each frequency band
        bands_mask = [
            np.logical_and(freqs >= fmin, freqs <= fmax) for fmin, fmax in FREQ_BANDS
        ]
        for i in range(len(FREQ_BANDS)):
            psd_masked = psds[:, bands_mask[i]]
            psds_mean = np.average(psd_masked, axis=1)
            psds_temp.append(psds_mean)
        PSDs.append(psds_temp)
    PSDs = np.array(PSDs)
    return PSDs


fmin = 4
fmax = 90
tmin = 0
tmax = 3
##Compute_PSD
for subj in SUBJ_LIST:
    for task in TASK_LIST:
        for run in RUN_LIST[task]:
            try:
                epochs_name, epochs_path = get_pareidolia_bids(
                    FOLDERPATH, subj, task, run, stage="epo_RS", cond=None
                )
                epochs_short = mne.read_epochs(epochs_path)

                epochs_name, epochs_path = get_pareidolia_bids(
                    FOLDERPATH, subj, task, run, stage="epo_RS", cond=None
                )
                epochs_long = mne.read_epochs(epochs_path)
                # Si vous voulez comparer les epochs entières (8sec), il est mieux de laisser de côté le début et la fin des epochs.
                psds_cond1, psds_cond2, freqs = compute_psd(
                    epochs_long,
                    FREQ_BANDS4,
                    psd_array_welch,
                    tmin=tmin,
                    tmax=tmax,
                    fmin=fmin,
                    fmax=fmax,
                    run=run,
                )
                print("FREQS ALL SHAPE: ", freqs.shape)
                print("PSDS ALL SHAPE: ", psds_cond1.shape)
                if run == "1":
                    psds_cond1 = np.average(
                        psds_cond1, axis=0
                    )  # remove empty dimension
                    slopes_cond = []
                    for elec in range(len(psds_cond1)):
                        print("ELEC: ", elec)
                        _, exp, _, _, _, _, _, _, _, _, slope_cond1 = compute_FOOOF(
                            freqs,
                            psds_cond1[elec],
                            1200,
                            precision=0.1,
                            max_freq=fmax,
                            min_freq=fmin,
                            noverlap=None,
                            nperseg=None,
                            nfft=None,
                            extended_returns=False,
                            graph=False,
                        )
                        print("SLOPE COND1: ", slope_cond1.shape)

                        slopes_cond.append(slope_cond1)

                if run == "2":
                    psds_cond2 = np.average(psds_cond2, axis=0)
                    slopes_cond = []
                    for elec in range(len(psds_cond2)):
                        print("ELEC: ", elec)
                        _, exp, _, _, _, _, _, _, _, _, slope_cond2 = compute_FOOOF(
                            freqs,
                            psds_cond2[elec],
                            1200,
                            precision=0.1,
                            max_freq=fmax,
                            min_freq=fmin,
                            noverlap=None,
                            nperseg=None,
                            nfft=None,
                            extended_returns=False,
                            graph=False,
                        )
                        print("SLOPE COND2: ", slope_cond2.shape)
                        slopes_cond.append(slope_cond2)

                PSDS = compute_bands(
                    epochs_short,
                    FREQ_BANDS4,
                    psd_array_welch,
                    tmin=tmin,
                    tmax=tmax,
                    slope_cond=slopes_cond,
                    fmin=fmin,
                    fmax=fmax,
                    run=run,
                )
                print("PSDS SHAPE: ", PSDS.shape)

                # le nom du stage doit commencer par PSD, la fin du nom est à votre choix
                psds_file, psds_path = get_pareidolia_bids(
                    FOLDERPATH,
                    subj,
                    task,
                    run,
                    stage="PSD_RS_corrected_3-90Hz",
                )
                savemat(psds_path, {"PSD": PSDS})
            except FileNotFoundError:
                pass
