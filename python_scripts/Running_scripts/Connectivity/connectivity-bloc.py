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


RUN_LIST = {'pareidolia':['1', '2', '3','4', '5', '6']}
#RUN_LIST = {'pareidolia':['1', '2']}
SUBJ_LIST = ['00', '01']
SUBJ_LIST = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
#SUBJ_LIST = ['07', '08', '09', '10', '11']
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

cond_names = ['Image_on_par', 'Image_on_nopar']
        


def connectivity(epochs, cond_names=None, run_list=None, method='pli', names=None, FOLDERPATH=None,
                tmin=0, tmax=None, sfreq=1200):
    
    if tmax is None:
        tmax = epochs.tmax
    epochs = epochs.crop(tmin, tmax)
    fmin, fmax = (2., 4., 8., 12., 20., 30., 50.), (4., 8., 12., 20, 30., 50., 70.)
    #fmin, fmax = (4., 8.), (8., 12.)

    #CHOICES OF METHODS: ['coh', 'cohy', 'imcoh', 'plv', 'ciplv', 'ppc', 'pli', 'wpli', 'wpli2_debiased']
    con_all = []
    for cond in cond_names:
        con = spectral_connectivity_epochs(
            epochs[cond], method=method, mode='multitaper', sfreq=sfreq, fmin=fmin, fmax=fmax,
            faverage=True, tmin=tmin, mt_adaptive=False, n_jobs=1, block_size=128)
        #for i in range(len(con.get_data(output='dense')[0, 0])):
        #    fig = plot_connectivity_circle(con.get_data(output='dense')[:, :, i], node_names=names, n_lines=100,
        #                fontsize_names=24, show=False, vmin=0.)
        #    fig[0].set_size_inches(24, 24)
        #    fig[0].savefig('connectivity_EPO_long_wpli_subj_'+subj+'_'+cond+'_'+str(i)+'.png')
        con_all.append(con)
    return con_all




par = []
nopar = []
method='wpli'
node_degree_par = []
node_degree_nopar = []
threshold=0.6
for subj in SUBJ_LIST:
    node_degree_par_run = []
    node_degree_nopar_run = []
    for r in RUN_LIST['pareidolia']:
        try:
            epochs_name, epochs_path = get_pareidolia_bids(FOLDERPATH, subj, 'pareidolia', r, stage = 'epo_long', cond=None)
            epochs = mne.read_epochs(epochs_path)
            con = connectivity(epochs, cond_names=cond_names, run_list=RUN_LIST['pareidolia'], FOLDERPATH=FOLDERPATH,
                                   names=names, method=method, tmin=0, tmax=1)
            par = con[0].get_data(output='dense')[:, :, :]
            nopar = con[1].get_data(output='dense')[:, :, :]
            np.save('connectivity_1sec_par_bloc_'+r+'subj_'+subj+'_'+method, par)
            np.save('connectivity_1sec_par_bloc_'+r+'subj_'+subj+'_'+method, nopar)
            node_deg_par = []
            node_deg_nopar = []
            for freq_band in range(len(par[0, 0])):
                node_deg_par.append(node_degree(par[:, :, freq_band], threshold=threshold))
                node_deg_nopar.append(node_degree(nopar[:, :, freq_band], threshold=threshold))
            node_degree_par_run.append(node_deg_par)
            node_degree_nopar_run.append(node_deg_nopar) 
        except (FileNotFoundError, KeyError):
            pass
    node_degree_par.append(node_degree_par_run)       
    node_degree_nopar.append(node_degree_nopar_run)          
            
            
np.save('node_degree_1sec_par_WPLI_blocs'+str(threshold), node_degree_par)
np.save('node_degree_1sec_nopar_WPLI_blocs'+str(threshold), node_degree_nopar)


'''for cond in cond_names:
    if cond == 'Image_on_par':
        data_ = par_connect
    if cond == 'Image_on_nopar':
        data_ = nopar_connect
    for i in range(len(data_[0, 0])):
        fig = plot_connectivity_circle(data_[:, :, i], node_names=names, n_lines=100,
                    fontsize_names=24, show=False)
        fig[0].set_size_inches(18, 18)
        fig[0].savefig('connectivity_EPO_long_wpli_ALL_subj_'+cond+'_'+str(i)+'.png')'''

            #bt_file, bt_path = get_pareidolia_bids(FOLDERPATH, subj, task, run, stage = 'bt_long_EMD_0.1')
            #biotuner_metrics.to_csv(bt_path, index=False)
