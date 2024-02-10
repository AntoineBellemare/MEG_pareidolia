import sys
sys.path.insert(0, 'C:/Users/Antoine/github/MEG_pareidolia/python_scripts/Functions')
import MEG_pareidolia_utils
from MEG_pareidolia_utils import *
from PARAMS import *


epochs_name, epochs_path = get_pareidolia_bids(FOLDERPATH, '00', 'pareidolia', '1', stage = 'epo_long', cond=None)
epochs = mne.read_epochs(epochs_path)
ch_xy = epochs.pick_types(meg=True, ref_meg=False).info


datapath = '../../../OUTPUTS/ML_accuracy_scores/'
filename = 'classifiers_epo_RT_before_allSpectral_allSpectral_100.csv'
data = pd.read_csv(datapath+filename)
classifiers = list(data['classifier'].unique())
participants = list(data['participant'].unique())

for participant in participants:
    for classifier in classifiers:
        value_to_plot_ = data.loc[data['participant'] == participant]
        print(classifier)
        value_to_plot = value_to_plot_.loc[value_to_plot_['classifier'] == classifier]
        print(value_to_plot)
        value_to_plot = list(value_to_plot['score'])
        p_vals = value_to_plot_.loc[value_to_plot_['classifier'] == classifier]['pvalue']
        _, p_vals = fdrcorrection(p_vals, alpha=0.05, method='indep')
        mask = p_values_boolean_1d(p_vals)
        #value_to_plot = list(value_to_plot[classifier])
        extreme = np.max((abs(np.min(np.min(np.array(value_to_plot)))), abs(np.max(np.max(np.array(value_to_plot)))))) # adjust the range of values
        vmax = extreme
        vmin = 0.5
        reportpath = 'ML_topomap_AllFeatures_{}_{}_.png'.format(participant, classifier)
        fig, ax = topoplot(value_to_plot, ch_xy, vmin=vmin, vmax=vmax, showtitle=True,
                                       mask = mask, figpath = reportpath, ax_title='decoding acc.')
