import mne  # Here we import mne, the package that will contain most of the function that we will use today.
from mne.datasets.brainstorm import (
    bst_raw,
)  # It is possible to import functions individually. This is helpful since it

# saves time, memory, and makes the calls to the function easier.
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs

# from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from scipy import stats
import pandas as pd
import os
from scipy.signal import welch
from fooof import FOOOF
from PARAMS import *
from statsmodels.stats.multitest import fdrcorrection
import sys
import collections

from collections.abc import Iterable


def get_empty_room_path(subject_id, base_dir):
    """
    Get the path to the empty room recording for a given subject.

    Parameters:
    - subject_id: str, the identifier for the subject.
    - base_dir: str, the base directory containing the BIDS data.

    Returns:
    - empty_room_path: str, path to the empty room recording.
    """
    # Create the path to the empty room recording directory based on the structure
    empty_room_dir_path = os.path.join(
        base_dir,
        f"sub-{subject_id}",
        "ses-NOISE",
        "meg",
        f"sub-{subject_id}_ses-NOISE_meg.ds",
    )

    # Check if the directory exists
    if not os.path.exists(empty_room_dir_path):
        print(f"Warning: Empty room recording for subject {subject_id} not found.")
        return None

    return empty_room_dir_path


def compute_noise_cov(subject_id, base_dir, l_freq=0.1, h_freq=150):
    """
    Compute the noise covariance matrix using empty room data.

    Parameters:
    - subject_id: str, the identifier for the subject.
    - base_dir: str, the base directory containing the BIDS data.
    - l_freq, h_freq: float, the frequency limits for filtering (default: 0.1 to 40 Hz).

    Returns:
    - noise_cov: Covariance matrix computed from the empty room data.
    """
    # Get the path to the empty room recording
    empty_room_path = get_empty_room_path(subject_id, base_dir)

    if empty_room_path is None:
        return None

    # Load the empty room recording into an MNE Raw object
    raw_empty_room = mne.io.read_raw_ctf(empty_room_path, preload=True)

    # Apply band-pass filtering
    raw_empty_room.filter(l_freq=l_freq, h_freq=h_freq)

    # Compute the noise covariance matrix
    noise_cov = mne.compute_raw_covariance(raw_empty_room, method="empirical")

    return noise_cov


def get_coregistration(FOLDERPATH, subj, fsaverage_dir):
    preproc_name, preproc_path = get_pareidolia_bids(
        FOLDERPATH, subj, "pareidolia", "1", stage="preproc", cond=None
    )
    raw = mne.io.read_raw_fif(preproc_path, preload=True)
    info = raw.info
    subject = "fsaverage"

    coreg = mne.coreg.Coregistration(
        info, subject, fsaverage_dir, fiducials="estimated"
    )
    coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose=True)
    coreg.omit_head_shape_points(distance=5.0 / 1000)
    coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=True)

    trans_path = op.join(
        base_directory,
        f"sub-{subj}",
        "ses-recording",
        "derivatives",
        f"sub-{subj}_ses-recording-trans.fif",
    )

    # Create the folder if it doesn't exist
    os.makedirs(op.dirname(trans_path), exist_ok=True)

    mne.write_trans(trans_path, coreg.trans, overwrite=True)

    return coreg


def savefig_cond(filename, fig):
    if not isinstance(fig, Iterable):
        fig = (fig,)
    for i, f in enumerate(fig):
        f.savefig(f"{filename}_{i}.jpg")


def array_topoplot(
    toplot,
    ch_xy,
    showtitle=False,
    titles=None,
    savefig=True,
    figpath=r"C:\Users\Dell\Jupyter\BrainHackSchool2019_AB\EEG_music_scripts",
    vmin=-1,
    vmax=1,
    mask=None,
    mask_marker_size=16,
):
    # create fig
    fig, ax = plt.subplots(1, len(toplot), figsize=(10, 5))
    # create a topomap for each data array
    for i, data in enumerate(toplot):
        image, _ = mne.viz.plot_topomap(
            data=data,
            pos=ch_xy,
            cmap="Purples",
            vmin=vmin,
            vmax=vmax,
            axes=ax[i],
            show=False,
            mask=mask[i, :],
        )

        # Adjust mask marker size
        for collection in ax[i].collections:
            if isinstance(collection, plt.collections.PathCollection):
                collection.set_sizes([mask_marker_size])  # Adjust marker size here
        # option for title
        if showtitle == True:
            ax[i].set_title(titles[i], fontdict={"fontsize": 10, "fontweight": "heavy"})
    # add a colorbar at the end of the line (weird trick from https://www.martinos.org/mne/stable/auto_tutorials/stats-sensor-space/plot_stats_spatio_temporal_cluster_sensors.html#sphx-glr-auto-tutorials-stats-sensor-space-plot-stats-spatio-temporal-cluster-sensors-py)
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax[-1])
    ax_colorbar = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(image, cax=ax_colorbar)
    ax_colorbar.tick_params(labelsize=8)
    # save plot if specified
    if savefig == True:
        plt.savefig(figpath, dpi=300)
    plt.show()
    return fig, ax


def compute_t_test(data1, data2):
    results = []
    for freq in range(data1.shape[1]):
        results_temp = []
        for elec in range(data1.shape[2]):
            data1_t_test = data1[:, freq, elec]
            data2_t_test = data2[:, freq, elec]
            results_temp.append(stats.ttest_rel(data1_t_test, data2_t_test))
        results.append(results_temp)
    results = np.array(results)
    t_values = results[:, :, 0]
    p_values = results[:, :, 1]
    return (results, t_values, p_values)


def compute_t_test_complex(data1, data2):
    results = []
    for elec in range(data1.shape[1]):
        data1_t_test = data1[:, elec]
        data2_t_test = data2[:, elec]
        results.append(stats.ttest_rel(data1_t_test, data2_t_test))
    results = np.array(results)
    t_values = results[:, 0]
    p_values = results[:, 1]
    return (results, t_values, p_values)


def p_values_boolean(p_values):
    p_values_boolean = p_values.copy()
    for e in range(p_values.shape[1]):
        for c in range(p_values.shape[0]):
            if p_values[c, e] < 0.05:
                p_values_boolean[c, e] = True
            else:
                p_values_boolean[c, e] = False
    p_values_boolean = np.array(p_values_boolean, dtype="bool")
    return p_values_boolean


def taskVSsham(subj, PSD_stage, EPO_stage, FOLDERPATH):
    RUN_LIST = {"pareidolia": ["1", "2", "3", "4", "5", "6", "7", "8"]}
    task = "pareidolia"
    task_PSD = []
    sham_PSD = []
    epochs_tot = []
    for e, run in enumerate(RUN_LIST[task]):
        try:
            PSD_name, PSD_path = get_pareidolia_bids(
                FOLDERPATH, subj, task, run, stage=PSD_stage, cond=None
            )
            epo_name, epo_path = get_pareidolia_bids(
                FOLDERPATH, subj, task, run, stage=EPO_stage, cond=None
            )
            epochs = mne.read_epochs(epo_path)
            epochs.pick_types(meg=True, ref_meg=False)
            epochs_data = epochs.get_data()
            print("HELLO")
            PSD = loadmat(PSD_path)
            PSD = PSD["PSD"]
            epochs_tot.append(epochs)
            if e < 6:
                task_PSD.append(PSD)
            else:
                try:
                    sham_PSD = PSD
                except:
                    pass
        except:
            pass

    task_PSD = np.array(task_PSD)
    sham_PSD = np.array(sham_PSD)
    return task_PSD, sham_PSD, epochs_tot


def match_n_trials(psds1, psds2):
    if len(psds1) > len(psds2):
        while len(psds1) > len(psds2):
            psds1 = np.delete(psds1, len(psds2), 0)
            # psds1 = np.array(psds1)
            # psds2 = np.array(psds2)
    if len(psds2) > len(psds1):
        while len(psds2) > len(psds1):
            psds2 = np.delete(psds2, len(psds1), 0)
            # psds2 = np.array(psds2)
            # psds1 = np.array(psds1)
    return (psds1, psds2)


def split_FD_anova(data, epochs):
    cond1 = []
    cond2 = []
    cond3 = []
    for e, i in enumerate(epochs.events[:, 2]):
        if i == 70 or i == 770 or i == 7770:
            cond1.append(data[e])
        if i == 71 or i == 771 or i == 7771:
            cond2.append(data[e])
        if i == 72 or i == 772 or i == 7772:
            cond3.append(data[e])
    cond1 = np.array(cond1)
    cond2 = np.array(cond2)
    cond3 = np.array(cond3)
    return cond1, cond2, cond3


def split_by_2_conditions(data, epochs, conditions, data_type="PSD"):
    c1_idx = []
    c2_idx = []
    for i, e in enumerate(epochs):
        if len(epochs[i][conditions[0]]) != 0:
            c1_idx.append(i)
        if len(epochs[i][conditions[1]]) != 0:
            c2_idx.append(i)
    cond1 = []
    cond2 = []

    if data_type == "PSD":
        for i1 in c1_idx:
            cond1.append(data[i1])
        for i2 in c2_idx:
            cond2.append(data[i2])
    if data_type == "Complexity":
        for i1 in c1_idx:
            cond1.append(data.loc[i1])
        for i2 in c2_idx:
            cond2.append(data.loc[i2])

    return np.array(cond1), np.array(cond2)


def split_by_2_conditions_meta(
    data, epochs, conditions, data_type="PSD", metadata=None
):
    c1_idx = []
    c2_idx = []
    for i, e in enumerate(epochs):
        if len(epochs[i][conditions[0]][metadata]) != 0:
            c1_idx.append(i)
        if len(epochs[i][conditions[1]][metadata]) != 0:
            c2_idx.append(i)
    cond1 = []
    cond2 = []

    if data_type == "PSD":
        for i1 in c1_idx:
            cond1.append(data[i1])
        for i2 in c2_idx:
            cond2.append(data[i2])
    if data_type == "Complexity":
        for i1 in c1_idx:
            cond1.append(data.loc[i1])
        for i2 in c2_idx:
            cond2.append(data.loc[i2])

    return np.array(cond1), np.array(cond2)


def array_topoplot(
    toplot,
    ch_xy,
    showtitle=False,
    titles=None,
    savefig=True,
    figpath=r"C:\Users\Dell\Jupyter\BrainHackSchool2019_AB\EEG_music_scripts",
    vmin=-1,
    vmax=1,
    mask=None,
):
    # create fig
    fig, ax = plt.subplots(1, len(toplot), figsize=(10, 5))
    # create a topomap for each data array
    for i, data in enumerate(toplot):
        image, _ = mne.viz.plot_topomap(
            data=data,
            pos=ch_xy,
            cmap="Spectral_r",
            vmin=vmin,
            vmax=vmax,
            axes=ax[i],
            show=False,
            mask=mask[i, :],
        )
        # option for title
        if showtitle == True:
            ax[i].set_title(titles[i], fontdict={"fontsize": 10, "fontweight": "heavy"})
    # add a colorbar at the end of the line (weird trick from https://www.martinos.org/mne/stable/auto_tutorials/stats-sensor-space/plot_stats_spatio_temporal_cluster_sensors.html#sphx-glr-auto-tutorials-stats-sensor-space-plot-stats-spatio-temporal-cluster-sensors-py)
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax[-1])
    ax_colorbar = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(image, cax=ax_colorbar)
    ax_colorbar.tick_params(labelsize=8)
    # save plot if specified
    if savefig == True:
        plt.savefig(figpath, dpi=300)
    # plt.show()
    return fig, ax


def compute_t_test(data1, data2):
    results = []
    for freq in range(data1.shape[1]):
        results_temp = []
        for elec in range(data1.shape[2]):
            data1_t_test = data1[:, freq, elec]
            data2_t_test = data2[:, freq, elec]
            results_temp.append(stats.ttest_rel(data1_t_test, data2_t_test))
        results.append(results_temp)
    results = np.array(results)
    t_values = results[:, :, 0]
    p_values = results[:, :, 1]
    return (results, t_values, p_values)


def get_pareidolia_bids(FOLDERPATH, subj, task, run, stage, cond=None):
    """
    Constructs BIDS basename and filepath in the SAflow database format.
    """
    if "meg" in stage:
        extension = ".ds"
    if "preproc" in stage:
        extension = "_raw.fif"
    if "epo" in stage:  # determine extension based on stage
        extension = ".fif"
    if "PSD" in stage:
        extension = ".mat"
    if "FOOOF" in stage:
        extension = ".csv"
    if "fig" in stage:
        extension = ".jpg"
    if "Morlet" in stage:
        extension = "-tfr.h5"
    if "report" in stage:
        extension = ".html"
    if "behav" in stage:
        extension = ".csv"
    if "spectral" in stage:
        extension = ".mat"
    elif "Complexity" in stage:
        extension = ".csv"
    elif "array_comp" in stage:
        extension = ".txt"
    elif "events" in stage:
        extension = ".tsv"
    elif "bt" in stage:
        extension = ".csv"
    elif "ARlog" in stage:
        extension = ".hdf5"
    elif "DFA" in stage:
        extension = ".csv"
    elif "Hilbert" in stage:
        extension = ".mat"
    elif "stc" in stage:
        extension = ".h5"

    if "events" in stage:
        pareidolia_bidsname = "sub-{}_ses-recording_task-{}_run-{}_{}{}".format(
            subj, task, run, stage, extension
        )
    if "stc" in stage:
        print(extension)
        pareidolia_bidsname = []
        pareidolia_bidspath = []
        for i in range(1, 53):
            # exmaple name : sub-07_ses-recording_task-pareidolia_run-1_epo-31-stc
            pareidolia_bidsname.append(
                "sub-{}_ses-recording_task-{}_run-{}_epo-{}-{}{}".format(
                    subj, task, run, i, stage, extension
                )
            )
            pareidolia_bidspath.append(
                os.path.join(
                    FOLDERPATH,
                    "sub-{}".format(subj),
                    "ses-recording",
                    "derivatives",
                    "source",
                    pareidolia_bidsname[i - 1],
                )
            )
    else:
        if cond == None:  # build basename with or without cond
            pareidolia_bidsname = "sub-{}_ses-recording_task-{}_run-{}_{}{}".format(
                subj, task, run, stage, extension
            )
        else:
            pareidolia_bidsname = "sub-{}_ses-recording_task-{}_run-{}_{}_{}{}".format(
                subj, task, run, cond, stage, extension
            )

        pareidolia_bidspath = os.path.join(
            FOLDERPATH,
            "sub-{}".format(subj),
            "ses-recording",
            "meg",
            pareidolia_bidsname,
        )
    return pareidolia_bidsname, pareidolia_bidspath


def reformat_events_meta(
    events,
    FDlist=None,
    RT_thresh=5000,
    task="pareidolia",
    run=None,
    n_objects_list=None,
):
    if task == "pareidolia":
        RT = []
        for e in range(len(events)):
            try:
                if events[e][2] == 4 and events[e + 1][2] == 6:
                    RT.append(events[e + 1][0] - events[e][0])
                while (
                    events[e][2] == 6 and events[e + 1][2] == 6
                ):  # remove events that correspond to pressing the response button more than one                                                               # time during a single trial
                    print("remove double press")
                    events = np.delete(events, (e + 1), axis=0)
                if (
                    events[e][2] == 6 and events[e + 2][2] == 6
                ):  # Remove button presses that happen after the end of the image
                    events = np.delete(events, (e + 2), axis=0)
                if (
                    events[e][2] == 5 and events[e + 1][2] == 6
                ):  # Remove button presses that happen after the end of the image
                    events = np.delete(events, (e + 1), axis=0)
            except:
                pass
        medianRT = np.median(RT)
        try:
            medianRT = int(medianRT)
        except:
            medianRT = 5000  # 4 seconds corresponds to middle of the trial

        print("medianRT", medianRT)
        m = -1
        for e in range(len(events)):
            try:
                if events[e][2] == 4:
                    m += 1
                    if (
                        events[e + 1][2] == 6 and not n_objects_list[m] >= 1
                    ):  # remove events when no response for number of object
                        print("remove event")
                        # events[e+1][2] = 666
                        events = np.delete(events, (e + 1), axis=0)
            except IndexError:
                pass
        # print('EVENTS', events)
        m = -1
        for e in range(len(events)):
            if events[e][2] == 4:
                m += 1

                # print('TEST-event', events[e+1][2])#, 'TEST-obj', n_objects_list[m])
                try:
                    if (
                        events[e + 1][2] != 6 and n_objects_list[m] >= 1
                    ):  # identity trials when no motor response but number of objects is reported
                        events[e][2] = 44
                        events = np.insert(
                            events, [e + 1], [events[e][0] + 100, 0, 666], axis=0
                        )
                except:
                    pass
                    # print('add response event')
                    # events = np.insert(events, e+1, np.array([events[e][0]+medianRT, 0, 6]), 0)
                    # print(events[e-3:e+3])
        for e in range(len(events)):
            try:
                if events[e][2] == 4 and events[e + 1][2] == 6:
                    events[e][2] = 44
            except IndexError:
                pass

    if task == "RS":
        if run == "1":
            events[:, 2] = np.where(events[:, 2] == 1, 100, events[:, 2])
            events[:, 2] = np.where(events[:, 2] == 2, 101, events[:, 2])
        if run == "2":
            events[:, 2] = np.where(events[:, 2] == 1, 200, events[:, 2])
            events[:, 2] = np.where(events[:, 2] == 2, 201, events[:, 2])

    return events, medianRT, RT


def remove_absent_event_ids(events, event_id):
    events = events[:, 2]
    print(events)
    keys2remove = []
    for e, key in zip(event_id.values(), event_id.keys()):
        if e not in events:
            print(e)
            keys2remove.append(key)

    for k in keys2remove:
        del event_id[k]

    return event_id


def FD2FDclass(FDlist):
    FDclass = FDlist.copy()
    for i in range(len(FDlist)):
        if FDlist[i] <= 1.3:
            FDclass[i] = 0
        if FDlist[i] >= 1.7:
            FDclass[i] = 2
        if FDlist[i] > 1.3 and FDlist[i] < 1.7:
            FDclass[i] = 1
    return FDclass


def arrange_dataframe(all_frame):
    # Create n_objets without zeros
    all_frame["positive_n_objets"] = all_frame["n_objets.response"]
    # Creates a boolean column that represents the presence or absence of pareidolia
    all_frame["boolean_n_objets"] = all_frame["n_objets.response"]
    all_frame.loc[all_frame["boolean_n_objets"] == "None", ["boolean_n_objets"]] = 0
    all_frame.loc[all_frame["boolean_n_objets"] == "2", ["boolean_n_objets"]] = "1"
    all_frame.loc[all_frame["boolean_n_objets"] == "3", ["boolean_n_objets"]] = "1"
    all_frame.loc[all_frame["boolean_n_objets"] == "4", ["boolean_n_objets"]] = "1"
    all_frame.loc[all_frame["boolean_n_objets"] == "5", ["boolean_n_objets"]] = "1"
    # Transforms 'none' in '0'
    all_frame.loc[all_frame["n_objets.response"] == "None", ["n_objets.response"]] = 0
    # Creates a 'bloc' column
    all_frame.loc[all_frame["trials_1.thisRepN"] == 0, ["trials.thisRepN"]] = 1
    all_frame.loc[all_frame["trials_2.thisRepN"] == 0, ["trials.thisRepN"]] = 1
    all_frame.loc[all_frame["trials_3.thisRepN"] == 0, ["trials.thisRepN"]] = 2
    all_frame.loc[all_frame["trials_4.thisRepN"] == 0, ["trials.thisRepN"]] = 2
    all_frame.loc[all_frame["trials_5.thisRepN"] == 0, ["trials.thisRepN"]] = 3
    all_frame.loc[all_frame["trials_6.thisRepN"] == 0, ["trials.thisRepN"]] = 3
    all_frame.loc[all_frame["trials_7.thisRepN"] == 0, ["trials.thisRepN"]] = 4
    all_frame.loc[all_frame["trials_8.thisRepN"] == 0, ["trials.thisRepN"]] = 4
    all_frame.loc[all_frame["trials_9.thisRepN"] == 0, ["trials.thisRepN"]] = 5
    all_frame.loc[all_frame["trials_10.thisRepN"] == 0, ["trials.thisRepN"]] = 5
    try:
        all_frame.loc[all_frame["trials_11.thisRepN"] == 0, ["trials.thisRepN"]] = 6
        all_frame.loc[all_frame["trials_12.thisRepN"] == 0, ["trials.thisRepN"]] = 6
    except:
        pass
    try:
        all_frame.loc[all_frame["trials_sham_1.thisRepN"] == 0, ["trials.thisRepN"]] = 7
        all_frame.loc[all_frame["trials_sham_2.thisRepN"] == 0, ["trials.thisRepN"]] = 7
        all_frame.loc[all_frame["trials_sham_3.thisRepN"] == 0, ["trials.thisRepN"]] = 8
        all_frame.loc[all_frame["trials_sham_4.thisRepN"] == 0, ["trials.thisRepN"]] = 8
    except:
        pass
    all_frame["bloc"] = all_frame["trials.thisRepN"]
    # Delete rows from practice bloc
    all_frame.rename(columns={"trials_practice.thisRepN": "practice"}, inplace=True)
    all_frame = all_frame.drop(all_frame[all_frame.practice == 0].index)
    # Change 'objects' to 'floats'
    all_frame["boolean_n_objets"] = all_frame["boolean_n_objets"].astype("float")
    all_frame["n_objets.response"] = all_frame["n_objets.response"].astype("float")

    all_frame.loc[:, "positive_n_objets"] = all_frame.loc[
        :, "positive_n_objets"
    ].replace("None", np.NaN)
    all_frame["positive_n_objets"] = all_frame["positive_n_objets"].astype("float")

    all_frame.loc[:, "reaction_time"] = all_frame.loc[:, "reaction_time.rt"]
    all_frame.loc[:, "reaction_time"] = all_frame.loc[:, "reaction_time"].clip(0, 8)

    return all_frame


def n_obj2class(n_obj):
    n_obj_class = []
    for i in n_obj:
        if i > 1:
            n_obj_class.append(2)
        if i == 1:
            n_obj_class.append(1)
        if i == 0:
            n_obj_class.append(0)
    return n_obj_class


def earlyVSlate(events, RT_thresh):
    earlyVSlate_vec = []
    i = 0
    for e in range(len(events[:, 2])):
        # Early
        if (
            events[e][2] == 44
            and events[e + 1][2] == 6
            and ((events[e + 1][0]) - (events[e][0])) <= RT_thresh
        ):
            earlyVSlate_vec.append(1)
            i += 1
        # Late
        elif (
            events[e][2] == 44
            and events[e + 1][2] == 6
            and ((events[e + 1][0]) - (events[e][0])) > RT_thresh
        ):
            earlyVSlate_vec.append(2)
            i += 1
        # None
        elif events[e][2] == 44:
            earlyVSlate_vec.append(100)
            i += 1
        if events[e][2] == 4:
            earlyVSlate_vec.append(0)
            i += 1
    print("this is I", i)
    return earlyVSlate_vec


def compute_t_test_complex(data1, data2):
    results = []
    for elec in range(data1.shape[1]):
        data1_t_test = data1[:, elec]
        data2_t_test = data2[:, elec]
        results.append(stats.ttest_rel(data1_t_test, data2_t_test))
    results = np.array(results)
    t_values = results[:, 0]
    p_values = results[:, 1]
    return (results, t_values, p_values)


def p_values_boolean_1d(p_values, threshold=0.05):
    p_values_boolean = p_values.copy()
    for e in range(len(p_values)):
        if p_values[e] < threshold:
            p_values_boolean[e] = True
        else:
            p_values_boolean[e] = False
    p_values_boolean = np.array(p_values_boolean, dtype="bool")
    return p_values_boolean


def p_values_boolean_complex(p_values, threshold=0.05):
    p_values_boolean = p_values.copy()
    for e in range(p_values.shape[0]):
        if p_values[e] < threshold:
            p_values_boolean[e] = True
        else:
            p_values_boolean[e] = False
    p_values_boolean = np.array(p_values_boolean, dtype="bool")
    return p_values_boolean


def topoplot(
    toplot,
    ch_xy,
    showtitle=False,
    titles=None,
    savefig=True,
    figpath=r"C:\Users\Dell\Jupyter\BrainHackSchool2019_AB\EEG_music_scripts",
    vmin=-1,
    vmax=1,
    ax_title="t values",
    mask=None,
    cmap="Spectral_r",
    mask_marker_size=10,
):
    # create fig
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # create a topomap for each data array
    image, _ = mne.viz.plot_topomap(
        data=toplot,
        pos=ch_xy,
        cmap=cmap,
        vlim=[vmin, vmax],
        outlines="head",
        sphere=0.19,
        axes=ax,
        show=False,
        mask=mask,
        mask_params={"markersize": mask_marker_size, "marker": "o"},
        contours=0,
    )

    # option for title
    if showtitle == True:
        ax.set_title(titles, fontdict={"fontsize": 10, "fontweight": "heavy"})
    # add a colorbar at the end of the line (weird trick from https://www.martinos.org/mne/stable/auto_tutorials/stats-sensor-space/plot_stats_spatio_temporal_cluster_sensors.html#sphx-glr-auto-tutorials-stats-sensor-space-plot-stats-spatio-temporal-cluster-sensors-py)
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    ax_colorbar = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(image, cax=ax_colorbar)
    ax_colorbar.set_title(ax_title)
    ax_colorbar.tick_params(labelsize=8)
    # save plot if specified
    if savefig == True:
        plt.savefig(figpath, dpi=300, transparent=True)
    # plt.show()
    return fig, ax


def combine_metadata_PSD_complexity(
    subj_list,
    run_list,
    task,
    epo_stage=None,
    cp_stage=None,
    psd_stage=None,
    save=False,
    AVG_mode=False,
    DAT_df=None,
    LZ=None,
    bt_stage=None,
    save_multi=False,
    FOOOF_path=None,
    complex_imports=None,
    FOOOF_stage=None,
    savename="_",
):
    MEG_atlas = {
        "CL": list(range(0, 24)),
        "FL": list(range(24, 57)),
        "OL": list(range(57, 76)),
        "PL": list(range(76, 97)),
        "TL": list(range(97, 131)),
        "CR": list(range(131, 153)),
        "FR": list(range(153, 186)),
        "OR": list(range(186, 204)),
        "PR": list(range(204, 226)),
        "TR": list(range(226, 259)),
        "CZ": list(range(259, 263)),
        "Fz": list(range(263, 266)),
        "OZ": list(range(266, 269)),
        "PZ": list(range(269, 270)),
    }
    MEG_regions = list(MEG_atlas.keys())
    region_lists = []
    for r in MEG_regions:
        region_lists.append([MEG_atlas[r]][0])
    if FOOOF_path is not None:
        FOOOF_df = pd.read_csv(FOOOF_path)
        try:
            FOOOF_df = FOOOF_df.drop(columns=["peaks_center", "amps", "width"])
        except:
            pass
    list_df_subj = []
    for s in subj_list:
        try:
            list_df_run = []
            for run in run_list:
                try:
                    epo_name, epo_path = get_pareidolia_bids(
                        FOLDERPATH, s, task, run, stage=epo_stage, cond=None
                    )
                    epochs = mne.read_epochs(epo_path)

                    if cp_stage is not None:
                        cp_name, cp_path = get_pareidolia_bids(
                            FOLDERPATH, s, task, run, stage=cp_stage, cond=None
                        )
                        complexity = pd.read_csv(cp_path)
                        complexity = complexity.rename(
                            columns={"Unnamed: 0": "trials", "Unnamed: 1": "electrodes"}
                        )
                        print("Complexity", complexity.columns)

                    if psd_stage is not None:
                        psd_name, psd_path = get_pareidolia_bids(
                            FOLDERPATH, s, task, run, stage=psd_stage, cond=None
                        )

                        PSD = loadmat(psd_path)
                        PSD = PSD["PSD"]
                        PSD = np.moveaxis(PSD, 1, 2)
                        trials_df = []
                        for i in range(len(PSD)):
                            df_ = (
                                pd.DataFrame(PSD[i])
                                .reset_index()
                                .rename(
                                    columns={
                                        "index": "electrodes",
                                        0: "delta",
                                        1: "theta",
                                        2: "alpha",
                                        3: "low_beta",
                                        4: "high_beta",
                                        5: "gamma1",
                                        6: "gamma2",
                                    }
                                )
                            )
                            df_["trials"] = i
                            if AVG_mode == "whole":
                                df_ = df_.groupby(by="trials").mean().reset_index()
                            # print(df_)
                            trials_df.append(df_)
                        PSD_df = pd.concat(trials_df)
                        print("PSD", PSD_df.columns)

                    if task == "pareidolia":
                        metadata = epochs.metadata
                        metadata = metadata.reset_index(drop=True)
                        metadata["trials"] = range(len(metadata))

                    # Get pareidolia scores to append on RS dataframe.
                    if task == "RS":
                        data = pd.read_csv(
                            "../../Merged_dataframes/df_ALL_metadata_MEG_sub00to11_epo_RT_ALL_new.csv"
                        )
                        print(data.columns)
                        # remove freq_range column
                        data = data.drop(columns=["freq_range"])
                        data_participant = (
                            data.groupby(by="participant").mean().reset_index()
                        )
                        data_participant = data_participant[
                            ["n_obj", "parei", "participant"]
                        ]
                        print(data_participant.columns)

                    print(psd_stage, cp_stage)
                    if psd_stage is not None and cp_stage is not None:
                        complexity["bloc"] = int(run)
                        complexity["participant"] = int(s)
                        print(
                            "OK_________________________________________________________"
                        )
                        if task == "pareidolia":
                            total_df_ = pd.merge(complexity, metadata, on="trials")
                            total_df = pd.merge(
                                total_df_, PSD_df, on=["trials", "electrodes"]
                            )
                        if task == "RS":
                            total_df = pd.merge(
                                complexity, PSD_df, on=["trials", "electrodes"]
                            )
                            print("TOTAL DF", total_df.columns)
                    if psd_stage is None and cp_stage is not None:
                        complexity["bloc"] = int(run)
                        complexity["participant"] = int(s)
                        if task == "pareidolia":
                            total_df = pd.merge(complexity, metadata, on="trials")
                        if task == "RS":
                            total_df = complexity

                    if psd_stage is not None and cp_stage is None:
                        PSD_df["bloc"] = int(run)
                        PSD_df["participant"] = int(s)
                        if task == "pareidolia":
                            total_df = pd.merge(PSD_df, metadata, on="trials")
                        if task == "RS":
                            total_df = PSD_df

                    if LZ is not None:
                        LZ_file, LZ_path = get_pareidolia_bids(
                            FOLDERPATH, s, task, run, stage="array_comp_LZ_RS"
                        )
                        LZ = np.load(LZ_path + ".npy")
                        LZ.shape

                        trials_df = []
                        for t in range(len(LZ)):
                            df_ = (
                                pd.DataFrame(LZ[t])
                                .reset_index()
                                .rename(columns={"index": "electrodes", 0: "LZ"})
                            )
                            df_["trials"] = t
                            # df_['bloc'] = bloc
                            # df_['participant'] = subj
                            trials_df.append(df_)
                        LZ_df = pd.concat(trials_df)
                        total_df = pd.merge(
                            total_df, LZ_df, on=["trials", "electrodes"]
                        )

                    if complex_imports is not None:
                        for names in complex_imports:
                            df_ = total_df
                            print(names[0])
                            total_df = import_complexity(
                                names[0], names[1], df_, task=task, subj=s, run=run
                            )

                    if bt_stage is not None:
                        try:
                            bt_file, bt_path = get_pareidolia_bids(
                                FOLDERPATH, s, task, run, stage=bt_stage
                            )
                            bt_df = pd.read_csv(bt_path)
                            total_df = pd.merge(
                                total_df, bt_df, on=["trials", "electrodes"]
                            )
                        except:
                            pass

                    if FOOOF_stage is not None:
                        # try:
                        FOOOF_file, FOOOF_path_ = get_pareidolia_bids(
                            FOLDERPATH, s, task, run, stage=FOOOF_stage
                        )
                        FOOOF_df = pd.read_csv(FOOOF_path_)

                        FOOOF_df = FOOOF_df.drop(
                            columns=[
                                # "peaks_center",
                                # "width",
                                # "amps",
                                # "detrend_psd",
                                "participant",
                                "bloc",
                            ]
                        )
                        FOOOF_df = FOOOF_df.rename(columns={"elec": "electrodes"})
                        print(FOOOF_df.columns)
                        total_df = pd.merge(
                            total_df, FOOOF_df, on=["trials", "electrodes"], suffixes=()
                        )
                    # except:
                    #    pass

                    list_df_run.append(total_df)
                except FileNotFoundError:
                    pass
            df_temp = pd.concat(list_df_run)
            list_df_subj.append(df_temp)
        except (FileNotFoundError, ValueError):
            pass
    df_final = pd.concat(list_df_subj)
    print(df_final.columns)
    if DAT_df is not None:
        df_final = pd.merge(df_final, DAT_df, on="participant")

    if FOOOF_path is not None:
        print("LENGTH BEFORE : ", len(df_final))
        df_final = pd.merge(
            df_final, FOOOF_df, on=["participant", "bloc", "trials", "electrodes"]
        )
        print("LENGTH WITH FOOOF : ", len(df_final))
    if task == "RS":
        df_final["bloc"] = df_final["bloc"] - 1
        df_final = pd.merge(df_final, data_participant, on="participant")
    if AVG_mode == "regions":
        for r_list, r_name in zip(region_lists, MEG_regions):
            for j in r_list:
                df_final.loc[df_final["electrodes"] == j, "region"] = r_name
        df_final = (
            df_final.groupby(by=["region", "trials", "participant", "bloc"])
            .mean()
            .reset_index()
        )
    if save is True:
        save_name = (
            "../../Merged_dataframes/df_ALL_metadata_MEG_sub{}to{}_{}_{}.csv".format(
                subj_list[0], subj_list[-1], epo_stage, savename
            )
        )
        df_final.to_csv(save_name, index=False)
    if save_multi is True:
        for e in range(0, len(total_df["electrodes"].unique())):
            df = total_df.loc[total_df["electrodes"] == e]
            df.to_csv(
                "12participants_MEG_electrodes_sub_" + str(e) + ".csv", index=False
            )
    return df_final


def node_degree(con_mat, threshold=0.6):
    thresh_node_degree = []
    for i in range(len(con_mat)):
        n_nodes = 0
        for j in range(len(con_mat[0])):
            if con_mat[i, j] > threshold:
                n_nodes += 1
        thresh_node_degree.append(n_nodes)
    return thresh_node_degree


def FOOOF_aperiodic(
    data,
    sf,
    precision=0.1,
    max_freq=80,
    min_freq=2,
    noverlap=None,
    nperseg=None,
    nfft=None,
    extended_returns=False,
    graph=False,
):
    if nperseg is None:
        mult = 1 / precision
        nfft = sf * mult
        nperseg = nfft
        noverlap = nperseg // 10
    freqs1, psd = welch(data, sf, nfft=nfft, nperseg=nperseg, noverlap=noverlap)
    fm = FOOOF(
        peak_width_limits=[precision * 2, 3], max_n_peaks=50, min_peak_height=0.3
    )

    freq_range = [(sf / len(data)) * 2, max_freq]
    fm.fit(freqs1, psd, freq_range)
    if graph is True:
        fm.report(freqs1, psd, freq_range)

    try:
        offset = fm.get_params("aperiodic_params")[0]
        exp = fm.get_params("aperiodic_params")[1]
    except:
        offset = "NaN"
        exp = "NaN"
    try:
        cf = [x[0] for x in fm.get_params("peak_params")]
        amp = [x[1] for x in fm.get_params("peak_params")]
        width = [x[2] for x in fm.get_params("peak_params")]
    except IndexError:
        cf = "NaN"
        amp = "NaN"
        width = "NaN"
    corrected_spectrum = fm.power_spectrum_ - fm._ap_fit
    r2 = fm.r_squared_
    return (
        offset,
        exp,
        cf,
        amp,
        width,
        fm.fooofed_spectrum_,
        fm.freq_range,
        fm.freq_res,
        corrected_spectrum,
        r2,
        fm._ap_fit,
    )


def merge_multi_GLM(
    path,
    n_electrodes=270,
    graph=False,
    ch_xy=None,
    savename="_",
    pval_thresh=0.01,
    FDR=True,
    vlim=None,
):
    df = pd.read_csv(path + "coefficients_electrode_0.csv")
    df = df.rename(columns={"Unnamed: 0": "fixed_effect"})
    fixed_eff_names = list(df["fixed_effect"])
    fixed_eff_names_mod = [a.replace(":", "_by_") for a in fixed_eff_names]
    pval_name = df.columns[-1]
    dict_final = dict.fromkeys(fixed_eff_names_mod)

    for effect, new_name in zip(fixed_eff_names, fixed_eff_names_mod):
        effects = []
        pvals = []
        print(new_name)
        for n in range(0, n_electrodes):
            df_ = pd.read_csv(path + "coefficients_electrode_" + str(n) + ".csv")
            df_ = df_.rename(columns={"Unnamed: 0": "fixed_effect"})
            # print(df_)
            effect_ = float(df_.loc[df_["fixed_effect"] == effect, "Estimate"])
            effects.append(effect_)
            pval = float(df_.loc[df_["fixed_effect"] == effect, pval_name])
            if pval < 0.01:
                print(effect, ":", n, "pval", pval, "size ", effect_)
            pvals.append(pval)
        dict_final[new_name] = [effects, pvals]
    if "DAT.z" in dict_final.keys():
        dict_final.pop("DAT.z")
    if "(Intercept)" in dict_final.keys():
        dict_final.pop("(Intercept)")
    # if "FD.z" in dict_final.keys():
    #    dict_final.pop("FD.z")
    if "I(FD.z^2)" in dict_final.keys():
        dict_final.pop("I(FD.z^2)")
    if "contrast" in dict_final.keys():
        dict_final.pop("contrast")
    if vlim is None:
        vmin = np.min([np.min(dict_final[effect][0]) for effect in dict_final.keys()])
        vmax = np.max([np.max(dict_final[effect][0]) for effect in dict_final.keys()])
        if np.abs(vmin) > np.abs(vmax):
            vmax = np.abs(vmin)
        else:
            vmin = -np.abs(vmax)
    else:
        vmin = vlim[0]
        vmax = vlim[1]
    if graph is True:
        for e, effect in enumerate(list(dict_final.keys())):
            value_to_plot = dict_final[effect][0]
            pvals = dict_final[effect][1]
            if FDR is True:
                _, pvals = fdrcorrection(pvals, alpha=pval_thresh, method="indep")
            mask = p_values_boolean_1d(pvals, threshold=pval_thresh)
            extreme = np.max(np.abs(value_to_plot))  # get the maximum absolute value
            # vmax = extreme
            # vmin = -extreme
            reportpath = path + savename + effect + ".png"

            print(reportpath)
            # image,_ = mne.viz.plot_topomap(data=value_to_plot, pos=ch_xy, cmap='Spectral_r', vmin=vmin, vmax=vmax, axes=None, show=True, mask = p_welch_multitaper)
            fig, ax = topoplot(
                value_to_plot,
                ch_xy,
                vmin=vmin,
                vmax=vmax,
                showtitle=True,
                mask=mask,
                figpath=reportpath,
                ax_title="fixed-effect estimate",
                cmap="RdBu_r",
            )
    return dict_final


def import_complexity(filename, name, df2merge, subj=None, task="pareidolia", run="1"):
    file, path = get_pareidolia_bids(FOLDERPATH, subj, task, run, stage=filename)
    cp = np.load(path + ".npy")
    trials_df = []
    for t in range(len(cp)):
        df_ = (
            pd.DataFrame(cp[t])
            .reset_index()
            .rename(columns={"index": "electrodes", 0: name})
        )
        df_["trials"] = t
        # df_['bloc'] = bloc
        # df_['participant'] = subj
        trials_df.append(df_)
    df = pd.concat(trials_df)
    total_df = pd.merge(df2merge, df, on=["trials", "electrodes"])
    return total_df
