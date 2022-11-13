import numpy as np


import sys
sys.path.insert(0, 'C:/Users/Antoine/github/MEG_pareidolia/python_scripts/Functions')
import MEG_pareidolia_utils
from MEG_pareidolia_utils import combine_metadata_PSD_complexity
import pandas as pd

subj_list = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
run_list = ['1', '2', '3', '4', '5', '6']
run_list = ['1', '2']
task = 'RS'


DAT_df = pd.read_csv('../../questionnaire_data/DAT_MEG.csv')
cp_imports = [['array_comp_hurst_RS_3sec', 'hurst'], ['array_comp_LZ_RS', 'LZ']]
#cp_imports = [['array_comp_LZ_RT_before', 'LZ']]
#cp_imports = [['array_comp_hurst', 'hurst']]
df_total = combine_metadata_PSD_complexity(subj_list, run_list, task,
                                           epo_stage='epo_RS_3sec', cp_stage='Complexity_fast_RS_3sec', psd_stage='PSD_RS_3sec ',
                                          save=True, AVG_mode='none', DAT_df=DAT_df, LZ=None, bt_stage=None,
                                          save_multi=False, complex_imports=cp_imports, savename='RS_3sec_hurst_LZ')

'''for e in range(0, 270):
    df = df_total.loc[df_total['electrodes'] == e]
    df.to_csv('12participants_epo_LONG_MEG_electrodes_elec_'+str(e)+'.csv', index=False)'''
