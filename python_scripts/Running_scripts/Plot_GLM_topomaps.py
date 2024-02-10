import numpy as np
import sys

sys.path.insert(0, "C:/Users/Antoine/github/MEG_pareidolia/python_scripts/Functions")
import MEG_pareidolia_utils
from MEG_pareidolia_utils import merge_multi_GLM, get_pareidolia_bids
from PARAMS import FOLDERPATH
import pandas as pd
import mne

GLM_name = "GLM_long_complexity_randdouble"
savename = "RT_fooofed_allpeaks_p01"

epochs_name, epochs_path = get_pareidolia_bids(
    FOLDERPATH, "00", "pareidolia", "1", stage="epo_long", cond=None
)
epochs = mne.read_epochs(epochs_path)
ch_xy = epochs.pick_types(meg=True, ref_meg=False).info
print(ch_xy)

path = "C:/Users/Antoine/github/MEG_pareidolia/R_data/results_20240206-201107/"

dict_final = merge_multi_GLM(
    path,
    n_electrodes=270,
    graph=True,
    ch_xy=ch_xy,
    savename=savename,
    pval_thresh=0.05,
    FDR=True,
    vlim=None,
)
