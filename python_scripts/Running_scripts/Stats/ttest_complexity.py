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
import copy
import random


RUN_LIST = {'pareidolia':['1','2','3','4', '5']}
SUBJ_LIST = ['00', '01', '02', '04', '05', '06', '08', '10', '11']
TASK_LIST = ['pareidolia']
COMPLEX_LIST = ['Hjorth', 'PEn', 'SVDEn', 'DiffEn', 'KFD', 'PFD', 'RR', 'SFD', 'SpEn']
CONDITIONS = [['parei==0', 'parei==1']]
#CONDITIONS = [['FD_class== 2', 'FD_class== 1'], ['FD_class== 1', 'FD_class== 0'], ['FD_class== 2', 'FD_class== 0']]

all_complex = []
for subj in SUBJ_LIST:
    subj_cp = []
    t_cond = []
    p_cond = []
    for task in TASK_LIST:
        task_cp = []
        sham_cp = []
        epochs_tot = {}
        for e, run in enumerate(RUN_LIST[task]):
            cp_name, cp_path = get_pareidolia_bids(FOLDERPATH, subj, task, run, stage = 'Complexity', cond=None)
            epo_name, epo_path = get_pareidolia_bids(FOLDERPATH, subj, task, run, stage = 'epo_long_meta', cond=None)
            epochs = mne.read_epochs(epo_path)
            epochs_data = epochs.get_data()
            complexity = pd.read_csv(cp_path)
            complexity = complexity.rename(columns={"Unnamed: 0": "trials", "Unnamed: 1": "electrodes"})
            complexity = complexity.set_index(['trials', 'electrodes'])          
            epochs_tot[run] = epochs
            if e < 6:
                task_cp.append(complexity)
            else:
                sham_cp = complexity
                
        for condition in CONDITIONS:
            for complex_name in COMPLEX_LIST:
                cond1_tot = []
                cond2_tot = []
                #Split by conditions for each of the 3 first blocs (task_PSD)
                for i in range(len(task_cp)):
                    epochs_id = str(i+1)
                    cond1, cond2 = split_by_2_conditions(complexity[complex_name], epochs_tot[epochs_id], condition, data_type='Complexity')

                    print(cond1.shape, cond2.shape)
                    cond1_tot.append(cond1)
                    cond2_tot.append(cond2)
                all_blocs1 = np.concatenate(cond1_tot)
                all_blocs2 = np.concatenate(cond2_tot)

                TTEST_LIST = [all_blocs1, all_blocs2]#, sham1, sham2]
                TTEST_LIST_str = ['all_blocs1', 'all_blocs2']#, 'sham1', 'sham2']

                #Determine the channels to use in the topomap
                epo_name, epo_path = get_pareidolia_bids(FOLDERPATH, subj, 'pareidolia', '1', stage = 'epo_long_meta', cond=None)
                epochs = mne.read_epochs(epo_path)
                ch_xy = epochs.pick_types(meg=True, ref_meg=False).info

                from itertools import combinations

                #This function take the two conditions matrices and makes them of equal size (equal number of trials for each condition)
                ttest1, ttest2 = match_n_trials(TTEST_LIST[0], TTEST_LIST[1])
                print(np.array(ttest1).shape)
                results_multitaper, t_multitaper, p_multitaper = compute_t_test_complex(ttest1, ttest2)
                #Initialize variables:
                '''pR = []
                #Choose number of permutations:
                p=1000
                ttest1_copy = copy.copy(ttest1)
                #Initialize permutation loop:
                for i in range(0,p):
                  #Shuffle one of the features:
                    random.shuffle(ttest1_copy)
                    #Computed permuted correlations and store them in pR:
                    _, t_multitaper_, _ = compute_t_test_complex(ttest1_copy, ttest2)
                    pR.append(t_multitaper)

                #Significance:
                p_multitaper = len(np.where(np.abs(pR)>=np.abs(t_multitaper_))[0])/p'''
                #Generate a matrix of boolean values (0 or 1) to determine if p-value is significant (alpha value can be changed in p_value_boolean function)
                #p_values are multiplied by the number of electrodes to correct for multiple comparisons (Bonferonni correction)
                p_welch_multitaper = p_values_boolean_complex(p_multitaper)
                #Low values indicate that the first element of t-test is lower than the second element
                value_to_plot = t_multitaper #t-values are plotted
                extreme = np.max((abs(np.min(np.min(np.array(value_to_plot)))), abs(np.max(np.max(np.array(value_to_plot)))))) # adjust the range of values
                vmax = extreme
                vmin = -extreme
                reportname, reportpath = get_pareidolia_bids(FOLDERPATH, subj, 'pareidolia', '-', stage = 'fig_ttest_complexity_'+str(condition)+'_'+complex_name)

                #image,_ = mne.viz.plot_topomap(data=value_to_plot, pos=ch_xy, cmap='Spectral_r', vmin=vmin, vmax=vmax, axes=None, show=True, mask = p_welch_multitaper)
                fig, ax = topoplot(value_to_plot, ch_xy, vmin=vmin, vmax=vmax, showtitle=True, titles=complex_name, mask = p_welch_multitaper, figpath = reportpath);

'''for e, c in enumerate(CONDITIONS):
    tvals = np.array(t_tot)[:, e, :, :]
    pvals = np.array(p_tot)[:, e, :, :]
    pvals_final = np.average(pvals, axis=0)
    tvals_final = np.average(tvals, axis=0)
    p_boolean = p_values_boolean(pvals_final)
    #Low values indicate that the first element of t-test is lower than the second element
    value_to_plot = tvals_final #t-values are plotted
    extreme = np.max((abs(np.min(np.min(np.array(value_to_plot)))), abs(np.max(np.max(np.array(value_to_plot)))))) # adjust the range of values
    vmax = extreme
    vmin = -extreme
    reportname, reportpath = get_pareidolia_bids(FOLDERPATH, '11', 'pareidolia', '-', stage = 'fig_ttest_long_H_contrast_AVG'+str(c))
    fig, ax = array_topoplot(value_to_plot, ch_xy, vmin=vmin, vmax=vmax, showtitle=True, titles=FREQ_NAMES, mask = p_boolean, figpath = reportpath)'''