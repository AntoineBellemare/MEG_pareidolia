from MEG_pareidolia_utils import *
import mne
from mne.datasets.brainstorm import bst_raw
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy.io as sio
from scipy.io import savemat, loadmat
from PARAMS import *
from scipy.stats import pearsonr
from statsmodels.stats.multitest import fdrcorrection

RUN_LIST = {'pareidolia':['1', '2','3','4', '5', '6']}
TASK_LIST = ['pareidolia']
SUBJ_LIST = ['00', '01', '02', '03', '04', '05', '06', '08', '10', '11']
COMPLEX_LIST = ['Hjorth', 'PEn', 'SVDEn', 'DiffEn', 'KFD', 'PFD', 'RR', 'SFD', 'SpEn']
COMPLEX_LIST = ['DiffEn']
FREQ_NAMES = ['delta', 'theta', 'alpha', 'beta1', 'beta2', 'gamma1', 'gamma2']
all_complex = []
pvals_tot = []
rvals_tot = []
for subj in SUBJ_LIST:
    subj_cp = []
    for task in TASK_LIST:

        epochs_tot = []
        task_PSD = []
        sham_PSD = []
        for e, run in enumerate(RUN_LIST[task]):
            try:
                
                epo_name, epo_path = get_pareidolia_bids(FOLDERPATH, subj, task, run, stage = 'epo_long_meta', cond=None)
                PSD_name, PSD_path = get_pareidolia_bids(FOLDERPATH, subj, task, run, stage = 'PSD_long_meta', cond=None)
                PSD = loadmat(PSD_path)
                PSD = PSD['PSD']
                epochs = mne.read_epochs(epo_path)
                epochs_data = epochs.get_data()
                epochs_tot.append(epochs)
                if e < 6:
                    task_PSD.append(PSD)
                else:
                    sham_PSD.append(PSD)
            except FileNotFoundError:
                pass
                
        pvals_freq = []
        rvals_freq = []
        
        
        
        for freq, freq_name in zip(range(len(PSD[0])), FREQ_NAMES):
            FD_list = []
            PSD_list = []
            #Split by conditions for each of the 3 first blocs (task_PSD)
            for i in range(len(task_PSD)): 
                FD_list.append(epochs_tot[i].metadata['FD'])
                PSD_list.append(task_PSD[i][:, freq, :])
                
            FD_list = np.concatenate(FD_list)
            #PSD_list = np.concatenate(PSD_list)
            corr_values = []
            p_values = []
            for elec in range(len(task_PSD[0][freq][0])):
                PSD_list_elec = []
                for i in range(len(PSD_list)):
                    PSD_list_elec.append(PSD_list[i][:, elec])
                PSD_list_elec = np.concatenate(PSD_list_elec)
                r_value, p_val = pearsonr(list(FD_list), list(PSD_list_elec))
                corr_values.append(r_value)
                p_values.append(p_val)
            
            
            
                
            pvals_freq.append(p_values)
            rvals_freq.append(corr_values)
            _, p_values = fdrcorrection(p_values, alpha=0.05, method='poscorr')           
            #Determine the channels to use in the topomap
            epo_name, epo_path = get_pareidolia_bids(FOLDERPATH, subj, 'pareidolia', '2', stage = 'epo_long_meta', cond=None)
            epochs = mne.read_epochs(epo_path)
            ch_xy = epochs.pick_types(meg=True, ref_meg=False).info
            
            mask = p_values_boolean_1d(p_values)

            value_to_plot = corr_values #t-values are plotted
            extreme = np.max((abs(np.min(np.min(np.array(value_to_plot)))), abs(np.max(np.max(np.array(value_to_plot)))))) # adjust the range of values
            vmax = 0.5
            vmin = -0.5
            reportname, reportpath = get_pareidolia_bids(FOLDERPATH, subj, 'pareidolia', '-', stage = 'fig_FREQS-FD_corr_FDR_corrected_'+freq_name)

            #image,_ = mne.viz.plot_topomap(data=value_to_plot, pos=ch_xy, cmap='Spectral_r', vmin=vmin, vmax=vmax, axes=None, show=True, mask = p_welch_multitaper)
            fig, ax = topoplot(value_to_plot, ch_xy, vmin=vmin, vmax=vmax, showtitle=True, titles=freq_name, mask = mask, figpath = reportpath, ax_title='pearson r');
    pvals_tot.append(pvals_freq)
    rvals_tot.append(rvals_freq)