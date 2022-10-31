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
SUBJ_LIST = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
#SUBJ_LIST = ['00']
TASK_LIST = ['RS']
which_complex = ['fast']
#which_complex = None
method = None
baseline = (-2.5, -1.5)
tmin = 0
tmax = 3
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
                if task != 'RS':
                    epochs = epochs.apply_baseline((-1, -0.5))
                epochs.pick_types(meg=True, ref_meg=False)
                epochs = epochs.crop(tmin, tmax)
                epochs_data = epochs.get_data()
                complex_tot = []
                for i in range(len(epochs_data)):
                    complex_trials = pd.DataFrame()
                    for j in range(len(epochs_data[0])):

                        results, info = nk.complexity(epochs_data[i][j], which=which_complex)
                        complex_trials = complex_trials.append(results)

                    complex_tot.append(complex_trials.reset_index())
                complex_tot = pd.concat(complex_tot, keys=range(len(epochs_data)))
                complex_file, complex_path = get_pareidolia_bids(FOLDERPATH, subj, task, run, stage = 'Complexity_fast_RS_3sec')
                complex_tot.to_csv(complex_path, index=True)
            except FileNotFoundError:
                pass
