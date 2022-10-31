import mne
from mne_connectivity import spectral_connectivity_epochs
from mne.datasets import sample
from mne_connectivity.viz import plot_sensors_connectivity
from MEG_pareidolia_utils import*
from PARAMS import *
from mne.datasets import sample
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne_connectivity import SpectralConnectivity
from mne_connectivity.viz import plot_connectivity_circle
from mne.viz import circular_layout


RUN_LIST = {'pareidolia':['1', '2', '3','4', '5', '6', '7', '8']}
SUBJ_LIST = ['08', '09', '10', '11']
SUBJ_LIST = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
TASK_LIST = ['pareidolia']


MEG_atlas = {'CL': list(range(0, 24)), 'FL': list(range(24, 57)), 'OL': list(range(57, 76)), 
             'PL': list(range(76, 97)), 'TL': list(range(97, 131)),
             'CR': list(range(131, 153)), 'FR': list(range(153, 186)), 'OR': list(range(186, 204)), 
             'PR': list(range(204, 226)), 'TR': list(range(226, 259)), 'CZ': list(range(259, 263)),
             'Fz': list(range(263, 266)), 'OZ': list(range(266, 269)), 'PZ': list(range(269, 270))
            }
MEG_regions = list(MEG_atlas.keys())
region_lists = []
for r in MEG_regions:
    region_lists.append([MEG_atlas[r]][0])
    
    
dict_names = {'Central-L':12, 'Frontal-L':41, 'Occi-L':67, 'Parietal-L':86, 'Temporal-L':111, 'Central-R':141,
              'Frontal-R':170, 'Occi-R':195, 'Parietal-R':215, 'Temporal=R':242}
edges = [0, 24, 57, 76, 97, 131, 153, 186, 204, 226, 259, 263, 266, 269, 270]
names = []
for i in range(270):
    if i in dict_names.values():
        names.append([name for name, idx in dict_names.items() if idx == i][0])
    if i in edges:
        names.append('Y')

    if i not in dict_names.values() and i not in edges:
        names.append('.')

cond_names = ['RT_par', 'RT_nopar']
        


def connectivity(subj, cond_names=None, run_list=None, method='pli', names=None, FOLDERPATH=None):
    all_epochs = []
    epochs_name, epochs_path = get_pareidolia_bids(FOLDERPATH, subj, 'pareidolia', '1', stage = 'epo_RT_post', cond=None)
    epochs_ = mne.read_epochs(epochs_path)
    dev_head_t = epochs_.info['dev_head_t']
    for run in run_list:
        try:
            epochs_name, epochs_path = get_pareidolia_bids(FOLDERPATH, subj, 'pareidolia', run, stage = 'epo_RT_post', cond=None)
            epochs = mne.read_epochs(epochs_path)
            epochs.info['dev_head_t'] = dev_head_t
            all_epochs.append(epochs)
        except FileNotFoundError:
            pass
    epochs = mne.concatenate_epochs(all_epochs)
    epochs = epochs.crop(1, 2.5)
    fmin, fmax = (4., 8., 12., 20., 30.), (8., 12., 20, 30., 50.)
    sfreq = 1200  # the sampling frequency
    tmin = 0.0  # exclude the baseline period
    #epochs.load_data().pick_types(meg='grad')  # just keep MEG and no EOG now

    #CHOICES OF METHODS: ['coh', 'cohy', 'imcoh', 'plv', 'ciplv', 'ppc', 'pli', 'wpli', 'wpli2_debiased']
    con_all = []
    for cond in cond_names:
        con = spectral_connectivity_epochs(
            epochs[cond], method=method, mode='multitaper', sfreq=sfreq, fmin=fmin, fmax=fmax,
            faverage=True, tmin=tmin, mt_adaptive=False, n_jobs=1, block_size=128)
        for i in range(len(con.get_data(output='dense')[0, 0])):
            fig = plot_connectivity_circle(con.get_data(output='dense')[:, :, i], node_names=names, n_lines=100,
                        fontsize_names=24, show=False)
            fig[0].set_size_inches(18, 18)
            fig[0].savefig('connectivity_subj_'+subj+'_'+cond+'_'+str(i)+'.png')
        con_all.append(con)
    return con_all






for subj in SUBJ_LIST:
    for task in TASK_LIST:

        con = connectivity(subj=subj, cond_names=cond_names, run_list=RUN_LIST[task], FOLDERPATH=FOLDERPATH,
                               names=names)

            #bt_file, bt_path = get_pareidolia_bids(FOLDERPATH, subj, task, run, stage = 'bt_long_EMD_0.1')
            #biotuner_metrics.to_csv(bt_path, index=False)
