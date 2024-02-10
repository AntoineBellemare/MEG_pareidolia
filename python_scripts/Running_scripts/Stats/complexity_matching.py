
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
from MEG_pareidolia_utils import *
from mne.io import read_raw_fif
import PARAMS
from PARAMS import *

RUN_LIST = {'pareidolia':['1', '2','3','4', '5', '6']}
SUBJ_LIST = ['00', '01', '02', '04', '05', '06', '08', '10', '11']
#SUBJ_LIST = ['05', '06']
TASK_LIST = ['pareidolia']
COMPLEX_LIST = ['Hjorth', 'SVDEn', 'DiffEn', 'KFD', 'PFD', 'RR', 'SFD', 'SpEn']
COMPLEX_LIST = ['PSDslope', 'ApEn', 'WPEn', 'MSE', 'Hurst', 'LZC']
#COMPLEX_LIST = ['DiffEn']
metadata_cond = 'parei == 0'
all_complex = []
pvals_tot = []
rvals_tot = []
for subj in SUBJ_LIST:
    subj_cp = []
    for task in TASK_LIST:
        task_cp = []
        sham_cp = []
        epochs_tot = []
        epo_idxs = []
        for e, run in enumerate(RUN_LIST[task]):
            try:
                cp_name, cp_path = get_pareidolia_bids(FOLDERPATH, subj, task, run, stage = 'Complexity_RT_before', cond=None)
                epo_name, epo_path = get_pareidolia_bids(FOLDERPATH, subj, task, run, stage = 'epo_RT', cond=None)
                epochs = mne.read_epochs(epo_path)
                #epochs.metadata = epochs.metadata.reset_index()
                epochs_data = epochs.get_data()
                epochs_data = epochs_data
                epochs = epochs[metadata_cond]
                epo_idx = list(epochs.metadata.reset_index(drop=True)['FD'].keys())
                complexity = pd.read_csv(cp_path)
                complexity = complexity.rename(columns={"Unnamed: 0": "trials", "Unnamed: 1": "electrodes"})
                complexity = complexity.set_index(['trials', 'electrodes'])  
                epochs_tot.append(epochs)
                epo_idxs.append(epo_idx)
                cp_idx = []
                for i in epo_idx:
                    cp_idx.append(complexity.iloc[complexity.index.get_level_values('trials') == i])
                #print(cp_idx)
                complexity = pd.concat((cp_idx))
                if e < 6:
                    task_cp.append(complexity)
                else:
                    sham_cp = complexity
            except FileNotFoundError:
                pass
                
        pvals_comp = []
        rvals_comp = []
        for complex_name in COMPLEX_LIST:
            FD_list = []
            complex_list = []
            #Split by conditions for each of the 3 first blocs (task_PSD)
            for i in range(len(task_cp)): 
                FD_list.append(epochs_tot[i].metadata['FD'])
                complex_list.append(task_cp[i][complex_name])
                
            FD_list = np.concatenate(FD_list)
            
            corr_values = []
            p_values = []
            for elec in range(len(task_cp[0][complex_name][epo_idxs[0][0]])):
                complex_list_elec = []
                for i in range(len(complex_list)):
                    complex_list_elec.append(complex_list[i][:, elec])
                complex_list_elec = np.concatenate(complex_list_elec)
                print(len(FD_list), len(complex_list_elec))
                r_value, p_val = pearsonr(list(FD_list), list(complex_list_elec))
                ##PERMUTATIONS
                #Initialize variables:
                pR = []
                #Choose number of permutations:
                p=1000
                FD_list_copy = copy.copy(list(FD_list))
                #Initialize permutation loop:
                for i in range(0,p):
                  #Shuffle one of the features:
                    random.shuffle(FD_list_copy)
                    #Computed permuted correlations and store them in pR:
                    pR.append(stats.pearsonr(FD_list_copy,list(complex_list_elec))[0])

                #Significance:
                p_val = len(np.where(np.abs(pR)>=np.abs(r_value))[0])/p
                corr_values.append(r_value)
                p_values.append(p_val)
                
            pvals_comp.append(p_values)
            rvals_comp.append(corr_values)
            #_, p_values = fdrcorrection(p_values, alpha=0.05, method='indep')           
            #Determine the channels to use in the topomap
            epo_name, epo_path = get_pareidolia_bids(FOLDERPATH, subj, 'pareidolia', '2', stage = 'epo_RT', cond=None)
            epochs = mne.read_epochs(epo_path)
            ch_xy = epochs.pick_types(meg=True, ref_meg=False).info
            
            mask = p_values_boolean_1d(p_values)
            print(mask)

            value_to_plot = corr_values #t-values are plotted
            extreme = np.max((abs(np.min(np.min(np.array(value_to_plot)))), abs(np.max(np.max(np.array(value_to_plot)))))) # adjust the range of values
            vmax = extreme
            vmin = -extreme
            reportname, reportpath = get_pareidolia_bids(FOLDERPATH, subj, 'pareidolia', '-', stage = 'fig_comp_corr_RT_perm_test_NOparei_'+complex_name)

            #image,_ = mne.viz.plot_topomap(data=value_to_plot, pos=ch_xy, cmap='Spectral_r', vmin=vmin, vmax=vmax, axes=None, show=True, mask = p_welch_multitaper)
            fig, ax = topoplot(value_to_plot, ch_xy, vmin=vmin, vmax=vmax, showtitle=True, titles=complex_name, mask = mask, figpath = reportpath, ax_title='pearson r');
    pvals_tot.append(pvals_comp)
    rvals_tot.append(rvals_comp)

for e, c in enumerate(COMPLEX_LIST):
    rvals = np.array(rvals_tot)[:, e, :]
    pvals = np.array(pvals_tot)[:, e, :]
    pvals_final = np.average(pvals, axis=0)
    rvals_final = np.average(rvals, axis=0)
    #_, pvals_final = fdrcorrection(pvals_final, alpha=0.05, method='indep') 
    p_boolean = p_values_boolean_1d(pvals_final)
    #Low values indicate that the first element of t-test is lower than the second element
    value_to_plot = rvals_final #t-values are plotted
    extreme = np.max((abs(np.min(np.min(np.array(value_to_plot)))), abs(np.max(np.max(np.array(value_to_plot)))))) # adjust the range of values
    vmax = extreme
    vmin = -extreme
    reportname, reportpath = get_pareidolia_bids(FOLDERPATH, '11', 'pareidolia', '-', stage = 'fig_comp_corr_RT_perm_test_NOparei_AVG_'+str(c))
    fig, ax = topoplot(value_to_plot, ch_xy, vmin=vmin, vmax=vmax, showtitle=True, titles=c, mask = p_boolean, figpath = reportpath)