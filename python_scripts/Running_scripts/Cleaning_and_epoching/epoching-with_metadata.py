import mne  # Here we import mne, the package that will contain most of the function that we will use today.
import sys

sys.path.insert(0, "C:/Users/Antoine/github/MEG_pareidolia/python_scripts/Functions")
import MEG_pareidolia_utils
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

# from autoreject import AutoReject
from MEG_pareidolia_utils import *
from mne.io import read_raw_fif
import PARAMS
from PARAMS import *

# Here you need to put the same FOLDERPATH as in the preprocessing script

# This variable determines which pareidolia trials will be judged as 'early' or 'late'

# Here you choose which runs and subjects you want to epoch
RUN_LIST = {"pareidolia": ["1", "2", "3", "4", "5", "6", "7", "8"], "RS": ["1", "2"]}
# RUN_LIST = {'pareidolia':['7']}
SUBJ_LIST = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]
# SUBJ_LIST = ['08']
TASK_LIST = ["pareidolia"]
window = "RT"


##EPOCHING
##This is the main EPOCHING function, which takes as input the subject, task, run, as well as a boolean value whether to include information about
##early vs late pareidolia (slowVSfast = True or False) and whether to include fractal
def epoching(subj, task, run, window=None, RT_thresh=3000):
    preproc_name, preproc_path = get_pareidolia_bids(
        FOLDERPATH, subj, task, run, stage="preproc"
    )
    preproc = read_raw_fif(preproc_path, preload=True)
    events = mne.find_events(preproc, shortest_event=1)
    ##Get Fractal dimension info from behavioral data
    if task == "pareidolia":
        behav_name, behav_path = get_pareidolia_bids(
            FOLDERPATH, subj, task, "1", stage="behav"
        )
        behav = pd.read_csv(behav_path)
        behav = arrange_dataframe(behav)

        behav = behav.loc[behav["bloc"] == int(run)]
        # print(behav)
        FDlist = list(np.array(behav["FD"]))
        Contrast = list(np.array(behav["Contrast"]))
        n_objects_list = list(np.array(behav["positive_n_objets"]))
        n_objects_list = np.nan_to_num(n_objects_list)
        n_obj_class = n_obj2class(n_objects_list)
        parei = [1 if x == 2 else x for x in n_obj_class]
        events, medianRT, RT = reformat_events_meta(
            events,
            FDlist=FDlist,
            RT_thresh=RT_thresh,
            task=task,
            run=run,
            n_objects_list=n_objects_list,
        )
    if task == "RS":
        FDlist = None

    # This line uses the function 'reformat_events', which you can find in EEG_pareidolia_utils, to add information about FD and slowVSfast paeidolia in the
    # event ids.

    print("EVENTS", events)
    print("LENGHT EVENTS", len(events))
    # print(events[:50])

    # Here is a CRUCIAL part of the function, which determines which part of the signal is use for each epoch, and for the baseline.
    if window == "RT2":
        tmin, tmax = -1.0, 2.5
        baseline = (-1.0, -0.5)
    elif window == "before":
        tmin, tmax = -2.5, 1.5
        baseline = (-2.5, -1.5)
    elif window == "RT":
        tmin, tmax = -2.5, -0.5
        baseline = (-2.5, -1.5)
    else:
        tmin, tmax = (
            -1.5,
            8,
        )  # Here we define the amount of time we want to keep before (tmin) and after (tmax) the event.
        baseline = (-1.5, 0)
    # Identification of channels of interest
    picks = mne.pick_types(
        preproc.info, meg=True, ref_meg=True, eeg=True, eog=True, stim=False
    )

    ##This whole section determines which event_id to choose depending on the task, and the values set for slowVSfast and FD.
    if task == "RS":
        if run == "1":
            event_id = {"RS10": 100}
        if run == "2":
            event_id = {"RS20": 200}

    if task == "pareidolia":
        event_id = {"Image_on_nopar": 4, "Image_on_par": 44}
        if window == "RT":
            for e in range(
                len(events)
            ):  # start epochs at median RT for non-pareidolai trials
                if events[e][2] == 4:
                    events[e][0] = events[e][0] + medianRT
                try:
                    if events[e][2] == 6 and (
                        (events[e][0] - events[e - 1][0]) < (1.5 * 1200)
                    ):
                        events[e][
                            2
                        ] == 666  # Replace trials when epoch position does not fit in the data size.
                        print("epochs window too early")
                    if events[e][2] == 6 and (
                        (events[e + 1][0] - events[e][0]) < (tmax - tmax)
                    ):
                        events[e][
                            2
                        ] == 666  # Replace trials when epoch position does not fit in the data size.
                        print("epochs window too late")
                except:
                    pass
            event_id = {"RT_nopar": 4, "RT_par": 6, "False_RT": 666}

        # Here we call the function that generates the epochs, using all the necessary information created earlier
        print(event_id)

        event_id = remove_absent_event_ids(events, event_id)
        epochs = mne.Epochs(
            preproc,
            events=events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            reject=None,
            preload=True,
            picks=picks,
        )
        print(epochs.selection)

        # Creates the metadata DataFrame
        new_metadata = pd.DataFrame(index=range(len(FDlist)))
        # arr = np.array(events) # Create mechanism to count occurrences of events
        # events_arr = arr[:, 2]
        # count_epo = np.bincount(events_arr)
        # n_trials = count_epo[44] + count_epo[4]
        new_metadata["FD"] = FDlist
        new_metadata["FD_class"] = FD2FDclass(FDlist)
        new_metadata["n_obj"] = n_objects_list
        new_metadata["n_obj_class"] = n_obj_class
        new_metadata["parei"] = parei
        new_metadata["contrast"] = Contrast

        earlyVSlate_vec = earlyVSlate(events, RT_thresh)
        spont_par = earlyVSlate(events, 2000)
        if len(epochs) == 51 and window == "RT":
            new_metadata = new_metadata[1:]
            earlyVSlate_vec = earlyVSlate_vec[1:]
            spont_par = spont_par[1:]
        if subj == "04" or subj == "05" or subj == "11" or subj == "02":
            if len(epochs) < len(new_metadata):
                new_metadata = new_metadata[0 : len(epochs)]
                earlyVSlate_vec = earlyVSlate_vec[0 : len(epochs)]
                spont_par = spont_par[0 : len(epochs)]
                print("LESS THAN 52 TRIALS IN THIS BLOC")

        new_metadata["earlyVSlate"] = earlyVSlate_vec
        new_metadata["spont_par"] = spont_par
        epochs.metadata = new_metadata

    if task == "RS":
        preproc_name, preproc_path = get_pareidolia_bids(
            FOLDERPATH, subj, task, run, stage="preproc"
        )
        preproc = read_raw_fif(preproc_path, preload=True)
        events = mne.find_events(preproc, shortest_event=1)
        len_epo = 3
        n_samples = 3 * 1200
        n_epos = int(events[0][0] / n_samples)
        e = 0
        for i in range(n_epos):
            events = np.insert(events, [i], [e, 0, 55], axis=0)
            e = e + n_samples
        events = np.delete(events, (-1), axis=0)
        event_id = {"RS": 55}
        tmin = 0
        tmax = 3
        # baseline = (-1, 0)
        epochs = mne.Epochs(
            preproc,
            events=events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            reject=None,
            preload=True,
            picks=picks,
        )
    # You can get rid of those two line (which perform autorejection of bad epochs) if your computer have difficulties
    print(epochs)
    # ar = AutoReject()
    # epochs= ar.fit_transform(epochs)
    return epochs


# This is a simple loop that iterates through our runs for each subject (depending on the subject chosen at the beginning of the script)
for i, subj in enumerate(SUBJ_LIST):
    for j, task in enumerate(TASK_LIST):
        for run_str in RUN_LIST[task]:
            if subj + run_str != "114":
                if subj + run_str != "027":
                    try:
                        # run_str = str(run+1)
                        ## 'Window = None' when you want epochs for the full trial size (8sec), 'window = 'RT'' when you want epochs around the response event
                        epochs = epoching(subj, task, run_str, window=window)
                        # In this line, the 'stage' value needs to begin with 'epo', and then you can add anything at the end to identify which epochs
                        # have been created. This will represent the end of the name of the epoched file.
                        epochs_file, epochs_path = get_pareidolia_bids(
                            FOLDERPATH, subj, task, run_str, stage="epo_RT_early3s_real"
                        )
                        epochs.save(epochs_path, overwrite=True)
                    except (FileNotFoundError, AttributeError):
                        pass
