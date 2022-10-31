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

RUN_LIST = {'pareidolia':['1', '2' ,'3', '4', '5', '6', '7', '8'], 'RS':['1', '2']}
SUBJ_LIST = ['08', '09', '10', '11']
SUBJ_LIST = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
TASK_LIST = ['RS']

#import pdb; pdb.set_trace

#This function is used to compute power values for each frequency bands on each epochs
def compute_psd(epochs, FREQ_BANDS, function=psd_welch, tmin = None, tmax = None):
    #epochs are cropped as desire (tmin could be before '0', ex: -1.5, depending on the values used during epoching)
    #epochs = epochs.apply_baseline((-1, -0.5))
    epochs.pick_types(meg=True, ref_meg=False)
    epochs = epochs.crop(tmin, tmax)
    print(epochs.get_data().shape)
    PSDs = []
    #This loop iterates for each epoch
    for t in range(len(epochs[:].get_data())):
        psds_temp = []
        picks = mne.pick_types(epochs.info, meg=True, ref_meg=False, eeg=False, eog=False, stim=False)
        for min_, max_ in FREQ_BANDS:
            psds, freqs = function(epochs[t], fmin=min_, fmax=max_, n_jobs=1, picks = picks)  #PSDs are calculated with this function, giving power values and corresponding frequency bins as output
            psds = 10. * np.log10(psds)   #PSDs values are transformed in log scale to compensate for the 1/f natural slope
            psds_mean = np.average(psds, axis=0) #Get rid of an empty dimension
            psds_mean = np.average(psds_mean, axis=1) #Average across bins to obtain one value for the entire frequency range
            psds_temp.append(psds_mean)
        PSDs.append(psds_temp)
    PSDs = np.array(PSDs)
    return PSDs

#We don't use this function for the moment.
'''def compute_PSD_bp(epochs, sf, epochs_length, f=None):
    EEG_chs_n = list(range(1,125))
    if f == None:
        f = [ [4, 8], [8, 12], [12, 20], [20, 30], [30, 60], [60, 90], [90, 120] ]
    # Choose MEG channels
    data = epochs.get_data(picks = EEG_chs_n) # On sort les data de l'objet MNE pour les avoir dans une matrice (un numpy array pour être précis)
    data = data.swapaxes(0,1).swapaxes(1,2) # On réarange l'ordre des dimensions pour que ça correspond à ce qui est requis par Brainpipe
    objet_PSD = feature.power(sf=int(sf), npts=int(sf*epochs_length), width=int((sf*epochs_length)/2), step=int((sf*epochs_length)/4), f=f, method='hilbert1') # La fonction Brainpipe pour créer un objet de calcul des PSD
    #data = data[:,0:960,:] # weird trick pour corriger un pb de segmentation jpense
    #print(data.shape)
    psds = objet_PSD.get(data)[0] # Ici on calcule la PSD !
    return psds'''

##Compute_PSD
for subj in SUBJ_LIST:
    for task in TASK_LIST:
        for run in RUN_LIST[task]:
            try:
                epochs_name, epochs_path = get_pareidolia_bids(FOLDERPATH, subj, task, run, stage = 'epo_RS_3sec', cond=None)
                epochs = mne.read_epochs(epochs_path)
                #Si vous voulez comparer les epochs entières (8sec), il est mieux de laisser de côté le début et la fin des epochs.
                psds_welch= compute_psd(epochs, FREQ_BANDS, psd_multitaper, tmin = 0, tmax = 3)
                #le nom du stage doit commencer par PSD, la fin du nom est à votre choix
                psds_file, psds_path = get_pareidolia_bids(FOLDERPATH, subj, task, run, stage = 'PSD_RS_3sec')
                savemat(psds_path, {'PSD': psds_welch})
            except FileNotFoundError:
                pass
