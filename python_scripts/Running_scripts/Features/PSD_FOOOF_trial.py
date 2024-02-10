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
TASK_LIST = ["pareidolia"]

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
    max_freq=90,
    min_freq=4,
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
    fm = FOOOF(peak_width_limits=[1, 7], max_n_peaks=6)

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
def compute_slope(epochs, FREQ_BANDS, function=psd_array_welch, tmin=None, tmax=None):
    # epochs are cropped as desire (tmin could be before '0', ex: -1.5, depending on the values used during epoching)
    # epochs = epochs.apply_baseline((-1, -0.5))
    epochs.pick_types(meg=True, ref_meg=False)
    epochs = epochs.crop(tmin, tmax)
    print(epochs.get_data().shape)
    PSDs = []
    # condition 1 is when metadata 'parei' = 0, condition 2 is when metadata 'parei' = 1
    slopes_all = []
    offsets_all = []
    exps_all = []
    r2s_all = []
    for t in range(len(epochs[:].get_data())):
        epochs_data = epochs[t].get_data()
        psds, freqs = function(
            epochs_data,
            sfreq=SFREQ,
            fmin=2,
            fmax=90,
            n_fft=NFFT,
            n_per_seg=NPERSEG,
            n_jobs=-1,
        )
        print("PSDS LONG SHAPE: ", psds.shape)
        print("FREQS LONG SHAPE: ", freqs.shape)
        # print min and max of freqs
        print("FREQS MIN: ", np.min(freqs))
        print("FREQS MAX: ", np.max(freqs))
        # remove empty dimension
        psds = np.average(psds, axis=0)
        slopes = []
        offsets = []
        exps = []
        r2s = []
        for elec in range(len(psds)):
            offset, exp, _, _, _, _, _, _, _, r2, slope = compute_FOOOF(
                freqs,
                psds[elec],
                1200,
                precision=0.1,
                max_freq=90,
                min_freq=4,
                noverlap=None,
                nperseg=None,
                nfft=None,
                extended_returns=False,
                graph=False,
            )
            offsets.append(offset)
            exps.append(exp)
            r2s.append(r2)
            slopes.append(slope)
        slopes_all.append(slopes)
        offsets_all.append(offsets)
        exps_all.append(exps)
        r2s_all.append(r2s)
    slopes_all = np.array(slopes_all)
    offsets_all = np.array(offsets_all)
    exps_all = np.array(exps_all)
    r2s_all = np.array(r2s_all)
    return slopes_all, exps_all, offsets_all, r2s_all, freqs


def compute_bands(
    epochs,
    FREQ_BANDS,
    function=psd_array_welch,
    tmin=None,
    tmax=None,
    slopes=None,
):
    # epochs are cropped as desire (tmin could be before '0', ex: -1.5, depending on the values used during epoching)
    # epochs = epochs.apply_baseline((-1, -0.5))
    epochs.pick_types(meg=True, ref_meg=False)
    epochs = epochs.crop(tmin, tmax)
    print(epochs.get_data().shape)
    PSDs = []
    # This loop iterates for each epoch
    for t, slope in zip(range(len(epochs[:].get_data())), slopes):
        psds_temp = []
        epochs_data = epochs[t].get_data()
        # select MEG channels
        psds, freqs = function(
            epochs_data,
            sfreq=1200,
            fmin=4,
            fmax=90,
            n_jobs=-1,
            n_fft=NFFT,
            n_per_seg=NPERSEG_short,
        )  # PSDs are calculated with this function, giving power values and corresponding frequency bins as output
        psds = 10.0 * np.log10(psds)
        # get rid of an empty dimension
        psds = np.average(psds, axis=0)
        print("PSDS SHORT SHAPE: ", psds.shape)
        for elec in range(len(psds)):
            psds[elec] = psds[elec] - slope[elec]
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


##Compute_PSD
for subj in SUBJ_LIST:
    for task in TASK_LIST:
        for run in RUN_LIST[task]:
            try:
                epochs_name, epochs_path = get_pareidolia_bids(
                    FOLDERPATH, subj, task, run, stage="epo_long", cond=None
                )
                epochs_short = mne.read_epochs(epochs_path)

                epochs_name, epochs_path = get_pareidolia_bids(
                    FOLDERPATH, subj, task, run, stage="epo_long", cond=None
                )
                epochs_long = mne.read_epochs(epochs_path)
                # Si vous voulez comparer les epochs entières (8sec), il est mieux de laisser de côté le début et la fin des epochs.
                slopes_all, exps, offsets, r2s, freqs = compute_slope(
                    epochs_long, FREQ_BANDS4, psd_array_welch, tmin=-1.5, tmax=0
                )
                print("SLOPES ALL SHAPE: ", slopes_all.shape)
                PSDS = compute_bands(
                    epochs_short,
                    FREQ_BANDS4,
                    psd_array_welch,
                    tmin=-1.5,
                    tmax=0,
                    slopes=slopes_all,
                )
                print("PSDS SHAPE: ", PSDS.shape)
                print("EXPS SHAPE: ", np.array(exps).shape)
                print("OFFSETS SHAPE: ", np.array(offsets).shape)
                print("R2S SHAPE: ", np.array(r2s).shape)
                # add empty dimension to exps and offsets to match PSDS shape
                exps = np.expand_dims(exps, axis=1)
                offsets = np.expand_dims(offsets, axis=1)
                r2s = np.expand_dims(r2s, axis=1)
                # append exps, offsets, r2s to PSDS array in the last dimension
                PSDS = np.append(PSDS, np.array(exps), axis=1)
                PSDS = np.append(PSDS, np.array(offsets), axis=1)
                PSDS = np.append(PSDS, np.array(r2s), axis=1)

                print("PSDS SHAPE wSlope: ", PSDS.shape)
                # le nom du stage doit commencer par PSD, la fin du nom est à votre choix
                psds_file, psds_path = get_pareidolia_bids(
                    FOLDERPATH,
                    subj,
                    task,
                    run,
                    stage="PSD_RT_trial_shortslope_corrected_all",
                )
                savemat(psds_path, {"PSD": PSDS})
                # create a dataframe with the exps, offsets and r2s
                # save the dataframe
                exp_file, exp_path = get_pareidolia_bids(
                    FOLDERPATH, subj, task, run, stage="FOOOF_exp", cond=None
                )
                # remove empty dimension
                exps = np.squeeze(exps)
                offsets = np.squeeze(offsets)
                r2s = np.squeeze(r2s)
                # create df (first dim is the trials, second dim is the electrodes)
                # Create a MultiIndex from all combinations of trials and elec
                multi_index = pd.MultiIndex.from_product(
                    [range(len(exps)), range(len(exps[0]))], names=["trials", "elec"]
                )

                # Flatten the arrays
                exp_flat = exps.flatten()
                offset_flat = offsets.flatten()
                r2_flat = r2s.flatten()

                # Create the DataFrame
                df = pd.DataFrame(
                    {"exp": exp_flat, "offset": offset_flat, "r2": r2_flat},
                    index=multi_index,
                ).reset_index()
                df["participant"] = subj
                df["bloc"] = run
                print(df)
                print("EXPS SHAPE: ", exps.shape)
                print("OFFSETS SHAPE: ", offsets.shape)
                print("R2S SHAPE: ", r2s.shape)
                df.to_csv(exp_path)
            except (FileNotFoundError, numpy.AxisError):
                pass
