from MEG_pareidolia_utils import *
import mne
from mne.datasets.brainstorm import bst_raw
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy.io as sio
from scipy.io import savemat, loadmat
import PARAMS
from PARAMS import *
from mlneurotools.stats import ttest_perm

RUN_LIST = {'pareidolia':['1','2','3', '4', '5', '6']}
SUBJ_LIST = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
#SUBJ_LIST = ['06']
TASK_LIST = ['pareidolia']

CONDITIONS_LIST = [['n_obj_class== 1', 'n_obj_class== 2'], ['n_obj_class== 0', 'n_obj_class== 2']]
CONDITIONS_LIST = [['earlyVSlate== 1', 'earlyVSlate== 2'], ['parei== 0', 'parei== 1'], ['n_obj_class== 1', 'n_obj_class== 2'], ['n_obj_class== 0', 'n_obj_class== 2']]

#metadata = ['contrast==0']
PSD_stage='PSD_RT'
EPO_stage = 'epo_RT'
#all_PSDs = []
t_tot = []
p_tot = []
for subj in SUBJ_LIST:
    #subj_PSD = []
    task_PSD, sham_PSD, epochs_tot = taskVSsham(subj, PSD_stage,EPO_stage , FOLDERPATH)
    print('task_PSD', task_PSD.shape)
    t_cond = []
    p_cond = []
    for c in CONDITIONS_LIST:
        try:
            cond1_tot = []
            cond2_tot = []
            #Split by conditions for each of the experimental blocs (task_PSD)
            for i in range(len(task_PSD)):
                epochs_id = str(i+1)
                print(i)
                #cond1, cond2= split_by_2_conditions_meta(task_PSD[i], epochs_tot[i], conditions = c, metadata=metadata)
                cond1, cond2= split_by_2_conditions(task_PSD[i], epochs_tot[i], conditions = c)
                if cond1.shape[0] != 0 and cond2.shape[0] != 0: 
                    cond1_tot.append(cond1)
                    cond2_tot.append(cond2)
            
            cond1_tot = np.array(cond1_tot, dtype=object)
            cond2_tot = np.array(cond2_tot, dtype=object)
            print(cond1_tot[0].shape) 
            all_blocs1 = np.concatenate(cond1_tot)
            all_blocs2 = np.concatenate(cond2_tot)
            
            TTEST_LIST = [all_blocs1, all_blocs2]#, sham1, sham2]
            TTEST_LIST_str = ['all_blocs1', 'all_blocs2']#, 'sham1', 'sham2']
            conditions_string = str(c)

            #Determine the channels to use in the topomap
            print('EPO_stage', EPO_stage)
            epo_name, epo_path = get_pareidolia_bids(FOLDERPATH, subj, 'pareidolia', '2', stage = EPO_stage, cond=None)
            epochs = mne.read_epochs(epo_path)
            ch_xy = epochs.pick_types(meg=True, ref_meg=False).info

            from itertools import combinations
            t_list = list(combinations(TTEST_LIST, 2))
            t_list_str = list(combinations(TTEST_LIST_str, 2))
            #try:
            for t in range(len(t_list)):
                #This function take the two conditions matrices and makes them of equal size (equal number of trials for each condition)
                ttest1, ttest2 = match_n_trials(t_list[t][0], t_list[t][1])
                print(np.array(ttest1).shape)

                t_perm = []
                p_perm = []
                for freq_band in range(len(ttest1[0, :, :])): 
                    tvals, pvals = ttest_perm(ttest1[:, freq_band, :], ttest2[:, freq_band, :], # cond1 = IN, cond2 = OUT
                        n_perm=1000,
                        n_jobs=8,
                        correction='maxstat',
                        paired=False,
                        two_tailed=True)
                    t_perm.append(tvals)
                    p_perm.append(pvals)
                t_perm = np.array(t_perm)
                p_perm = np.array(p_perm)
                t_cond.append(t_perm)
                p_cond.append(p_perm)
                #results_multitaper, t_multitaper, p_multitaper = compute_t_test(ttest1, ttest2)
                #Generate a matrix of boolean values (0 or 1) to determine if p-value is significant (alpha value can be changed in p_value_boolean function)
                #p_values are multiplied by the number of electrodes to correct for multiple comparisons (Bonferonni correction)
                p_boolean = p_values_boolean(p_perm)
                #Low values indicate that the first element of t-test is lower than the second element
                value_to_plot = t_perm #t-values are plotted
                extreme = np.max((abs(np.min(np.min(np.array(value_to_plot)))), abs(np.max(np.max(np.array(value_to_plot)))))) # adjust the range of values
                vmax = extreme
                vmin = -extreme
                reportname, reportpath = get_pareidolia_bids(FOLDERPATH, subj, 'pareidolia', '-', stage = 'fig_ttest_RT_'+str(c))
                fig, ax = array_topoplot(value_to_plot, ch_xy, vmin=vmin, vmax=vmax, showtitle=True, titles=FREQ_NAMES, mask = p_boolean,                                            figpath = reportpath);
        except (TypeError, ZeroDivisionError): # When one condition is empty
            pass
    t_tot.append(t_cond)
    p_tot.append(p_cond)
    
for e, c in enumerate(CONDITIONS_LIST):
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
    reportname, reportpath = get_pareidolia_bids(FOLDERPATH, '11', 'pareidolia', '-', stage = 'fig_ttest_RT_AVG'+str(c))
    fig, ax = array_topoplot(value_to_plot, ch_xy, vmin=vmin, vmax=vmax, showtitle=True, titles=FREQ_NAMES, mask = p_boolean, figpath = reportpath)




