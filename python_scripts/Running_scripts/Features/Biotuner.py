import mne
from mne.datasets.brainstorm import bst_raw
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy.io as sio
#import brainpipe
#from brainpipe import feature
import pandas as pd
import scipy.io as sio
from scipy.io import savemat, loadmat
from MEG_pareidolia_utils import *
import PARAMS
from PARAMS import *
from biotuner.biotuner_utils import *
from biotuner.biotuner_object import *
from biotuner.metrics import *

RUN_LIST = {'pareidolia':['1', '2', '3','4', '5', '6', '7', '8']}
SUBJ_LIST = ['08', '09', '10', '11']
SUBJ_LIST = ['00', '01', '02', '03']
TASK_LIST = ['pareidolia']

#import pdb; pdb.set_trace


#This function is used to compute power values for each frequency bands on each epochs
def compute_consonance(epochs, peaks_function='EMD', precision=0.5, tmin = None, tmax = None):
    FREQ_BANDS = [[1, 3], [3, 7], [7, 12], [12, 18], [18, 30], [30, 45]] # Define frequency bands for peaks_function = 'fixed'
    #epochs are cropped as desire (tmin could be before '0', ex: -1.5, depending on the values used during epoching)
    epochs = epochs.apply_baseline((-1.5, 0))
    epochs.pick_types(meg=True, ref_meg=False)
    epochs = epochs.crop(tmin, tmax)
    print(epochs.get_data().shape)
    all_data = epochs.get_data()
    #This loop iterates for each epoch
    all_dfs = []
    for trial in range(len(all_data)):
        elec_dfs = []
        for electrode in range(len(all_data[0])):
            biotuning = compute_biotuner(sf = 1200, peaks_function = peaks_function, precision = precision, n_harm = 10,
                                ratios_n_harms = 5, ratios_inc_fit = False, ratios_inc = False) # Initialize biotuner object

            biotuning.peaks_extraction(all_data[trial, electrode, :], FREQ_BANDS = FREQ_BANDS, ratios_extension = False, max_freq = 80, n_peaks=5,
                                      graph=False, min_harms=2)
            biotuning.compute_peaks_metrics(n_harm=10, delta_lim=20)

            tun_metrics = tuning_to_metrics(biotuning.peaks_ratios)
            tun_metrics.update(biotuning.peaks_metrics)
            print(biotuning.peaks_metrics)
            del tun_metrics["harm_pos"]
            del tun_metrics["common_harm_pos"]
            df_cons = pd.DataFrame(tun_metrics, index=[0])
            df_cons['electrodes'] = electrode
            df_cons['trials'] = trial
            elec_dfs.append(df_cons)
        df_temp = pd.concat(elec_dfs)
        all_dfs.append(df_temp)
    df_final = pd.concat(all_dfs)
        
    return df_final



##Compute_PSD
for subj in SUBJ_LIST:
    for task in TASK_LIST:
        for run in RUN_LIST[task]:
            try:
                epochs_name, epochs_path = get_pareidolia_bids(FOLDERPATH, subj, task, run, stage = 'epo_long', cond=None)
                epochs = mne.read_epochs(epochs_path)
                #Si vous voulez comparer les epochs entières (8sec), il est mieux de laisser de côté le début et la fin des epochs.
                biotuner_metrics= compute_consonance(epochs, peaks_function='EMD', precision=0.1, tmin=0, tmax=8)
                #le nom du stage doit commencer par PSD, la fin du nom est à votre choix
                bt_file, bt_path = get_pareidolia_bids(FOLDERPATH, subj, task, run, stage = 'bt_long_EMD_0.1_sub')
                biotuner_metrics.to_csv(bt_path, index=False)
            except FileNotFoundError:
                pass
