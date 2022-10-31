import sys
sys.path.insert(0, 'C:/Users/Antoine/github/MEG_pareidolia/python_scripts/Functions')
import MEG_pareidolia_utils
from MEG_pareidolia_utils import *
import mne # Here we import mne, the package that will contain most of the function that we will use today.
from mne.datasets.brainstorm import bst_raw # It is possible to import functions individually. This is helpful since it
                                            # saves time, memory, and makes the calls to the function easier.
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
import time
import numpy as np
import scipy.io as sio
from scipy.io import savemat, loadmat
import nolds
import neurokit2 as nk
from scipy import stats
import matplotlib.pyplot as plt
from PARAMS import *
from neurokit2.complexity.complexity_lempelziv import complexity_lempelziv


RUN_LIST = {'pareidolia':['1','2','3','4', '5', '6', '7', '8'], 'RS':['1', '2']}
SUBJ_LIST = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
TASK_LIST = ['RS']
method = 'LZ'
baseline = (-1.5, 0)
tmin = 0
tmax = 8
delay=1
dimension=2
tolerance="default"
##Compute complexity
#https://github.com/neuropsychology/NeuroKit/blob/master/neurokit2/complexity/complexity.py

for subj in SUBJ_LIST:
    for task in TASK_LIST:
        for run in RUN_LIST[task]:
            try:
                epochs_name, epochs_path = get_pareidolia_bids(FOLDERPATH, subj, task, run, stage = 'epo_RS_3sec', cond=None)
                epochs = mne.read_epochs(epochs_path)
                epochs = epochs.apply_baseline()
                epochs.pick_types(meg=True, ref_meg=False)
                epochs = epochs.crop(tmin, tmax)
                epochs_data = epochs.get_data()
                complex_tot = []
                for i in range(len(epochs_data)):
                    complex_trials = []
                    for j in range(len(epochs_data[0])):

                        if method == 'LZ':
                            results, info = complexity_lempelziv(epochs_data[i][j], dimension=dimension, delay=delay,
                                                                         permutation=True)
                            print(results)
                        if method == 'MSE':
                            results, info = nk.entropy_multiscale(epochs_data[i][j], show=True, composite=True)
                            print(results)
                        if method == 'hurst':
                            results, info = nk.complexity_hurst(epochs_data[i][j])
                            print(results)
                        complex_trials.append(results)
                    complex_tot.append(complex_trials)
                complex_tot = np.array(complex_tot)
                complex_file, complex_path = get_pareidolia_bids(FOLDERPATH, subj, task, run, stage = 'array_comp_LZ_RS')
                np.save(complex_path, complex_tot)
            except FileNotFoundError:
                pass
