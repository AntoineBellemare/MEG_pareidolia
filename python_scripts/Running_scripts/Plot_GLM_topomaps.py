import numpy as np
import sys
sys.path.insert(0, 'C:/Users/Antoine/github/MEG_pareidolia/python_scripts/Functions')
import MEG_pareidolia_utils
from MEG_pareidolia_utils import merge_multi_GLM, get_pareidolia_bids
from PARAMS import FOLDERPATH
import pandas as pd
import mne

GLM_name = 'GLM_RT_before_ints'
savename = 'RT_before_ints'

epochs_name, epochs_path = get_pareidolia_bids(FOLDERPATH, '00', 'pareidolia', '1', stage = 'epo_long', cond=None)
epochs = mne.read_epochs(epochs_path)
ch_xy = epochs.pick_types(meg=True, ref_meg=False).info


path = 'C:/Users/Antoine/github/MEG_pareidolia/R_data/results/'+GLM_name

dict_final = merge_multi_GLM(path, n_electrodes=270, graph=True, ch_xy=ch_xy, savename=savename)
