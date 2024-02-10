import sys
sys.path.insert(0, 'C:/Users/Antoine/github/MEG_pareidolia/python_scripts/Functions')
import MEG_pareidolia_utils
from MEG_pareidolia_utils import *
from PARAMS import *


epochs_name, epochs_path = get_pareidolia_bids(FOLDERPATH, '00', 'pareidolia', '1', stage = 'epo_long', cond=None)
epochs = mne.read_epochs(epochs_path)
ch_xy = epochs.pick_types(meg=True, ref_meg=False).info


datapath = '../../../OUTPUTS/ML_accuracy_scores/'
filename = 'classifiers_epo_RT_before_StratGroupKfold_BAcc_minObs30_single_feature_LZ_100.csv'
data = pd.read_csv(datapath+filename)
classifiers = list(data['classifier'].unique())
participants = list(data['participant'].unique())
features = list(data['feature'].unique())

for participant in participants:
    for classifier in classifiers:
        for feature in features:
            value_to_plot_ = data.loc[data['participant'] == participant]
            print(classifier)
            value_to_plot_ = value_to_plot_.loc[value_to_plot_['classifier'] == classifier]
            value_to_plot_ = value_to_plot_.loc[value_to_plot_['feature'] == feature]
            value_to_plot = list(value_to_plot_['score'])
            p_vals = list(value_to_plot_['pvalue'])
            _, p_vals = fdrcorrection(p_vals, alpha=0.05, method='indep')
            mask = p_values_boolean_1d(p_vals)
            #value_to_plot = list(value_to_plot[classifier])
            extreme = np.max((abs(np.min(np.min(np.array(value_to_plot)))), abs(np.max(np.max(np.array(value_to_plot)))))) # adjust the range of values
            vmax = 0.75
            vmin = 0.5
            reportpath = '../../../OUTPUTS/ML_topomaps/ML_topomap_AllFeatures_Stratroupkfold_{}_{}_{}.png'.format(participant, classifier, feature)
            fig, ax = topoplot(value_to_plot, ch_xy, vmin=vmin, vmax=vmax, showtitle=True,
                                           mask = mask, figpath = reportpath, ax_title='decoding acc.')
