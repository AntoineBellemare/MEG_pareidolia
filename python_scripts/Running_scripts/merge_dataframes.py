import numpy as np


import sys

sys.path.insert(0, "C:/Users/Antoine/github/MEG_pareidolia/python_scripts/Functions")
import MEG_pareidolia_utils
from MEG_pareidolia_utils import combine_metadata_PSD_complexity
import pandas as pd

subj_list = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]
run_list = ["1", "2", "3", "4", "5", "6"]
# run_list = ["7", "8"]
# run_list = ["1", "2"]
task = "pareidolia"
FOOOF_path = "C:/Users/Antoine/github/MEG_pareidolia/Notebooks/FOOOF_long.csv"

DAT_df = pd.read_csv("../../questionnaire_data_perso/DAT_MEG.csv")
cp_imports = [
    ["array_comp_hurst_RS_3sec", "hurst"],
    ["array_comp_LZ_RS", "LZ"],
    ["array_comp_LZnp_RT", "LZ_smallperm"],
]
cp_imports = [["array_comp_LZnp_RT", "LZ"]]
# cp_imports = [['array_comp_hurst', 'hurst']]
df_total = combine_metadata_PSD_complexity(
    subj_list,
    run_list,
    task,
    epo_stage="epo_long",
    cp_stage=None,
    psd_stage="PSD_long",
    save=True,
    AVG_mode="none",
    DAT_df=DAT_df,
    bt_stage=None,
    save_multi=False,
    complex_imports=None,
    savename="long_FOOOF_exp",
    FOOOF_path=None,
    FOOOF_stage="FOOOF_exp",
)

"""for e in range(0, 270):
    df = df_total.loc[df_total['electrodes'] == e]
    df.to_csv('12participants_epo_LONG_MEG_electrodes_elec_'+str(e)+'.csv', index=False)"""
