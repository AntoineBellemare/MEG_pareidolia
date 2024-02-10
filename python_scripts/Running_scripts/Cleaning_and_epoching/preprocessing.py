import os
import numpy as np
import mne
from scipy.io import loadmat, savemat
from mne.io import read_raw_egi, read_raw_ctf
import matplotlib.pyplot as plt
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
import sys

sys.path.insert(0, "C:/Users/Antoine/github/MEG_pareidolia/python_scripts/Functions")
import MEG_pareidolia_utils
from MEG_pareidolia_utils import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from matplotlib.pyplot import close


def pareidolia_preproc(filepath, preprocpath, reportpath):
    report = mne.Report(verbose=True)
    raw_data = read_raw_ctf(filepath, preload=True)
    raw_data = raw_data.apply_gradient_compensation(
        grade=3
    )  # required for source reconstruction
    picks = mne.pick_types(raw_data.info, meg=True, eog=True, exclude="bads")
    fig = raw_data.plot(show=False)
    # report.add_figs_to_section(fig, captions="Time series", section="Raw data")
    close(fig)
    fig = raw_data.plot_psd(average=False, picks=picks, show=False)
    # report.add_figs_to_section(fig, captions="PSD", section="Raw data")
    close(fig)

    ## Filtering
    high_cutoff = 200
    low_cutoff = 0.5
    raw_data.filter(low_cutoff, high_cutoff, fir_design="firwin")
    raw_data.notch_filter(
        np.arange(60, high_cutoff + 1, 60),
        picks=picks,
        filter_length="auto",
        phase="zero",
        fir_design="firwin",
    )
    fig = raw_data.plot_psd(average=False, picks=picks, fmax=120, show=False)
    # report.add_figs_to_section(fig, captions="PSD", section="Filtered data")
    close(fig)

    ## ICA
    ica = ICA(n_components=20, random_state=0).fit(raw_data, decim=3)
    fig = ica.plot_sources(raw_data, show=False)
    # report.add_figs_to_section(fig, captions="Independent Components", section="ICA")
    close(fig)

    ## FIND ECG COMPONENTS
    ecg_threshold = 0.50
    ecg_epochs = create_ecg_epochs(raw_data, ch_name="EEG059")
    ecg_inds, ecg_scores = ica.find_bads_ecg(
        ecg_epochs, ch_name="EEG059", method="ctps", threshold=ecg_threshold
    )
    fig = ica.plot_scores(ecg_scores, ecg_inds, show=False)
    # report.add_figs_to_section(
    #    fig, captions="Correlation with ECG (EEG059)", section="ICA - ECG"
    # )
    close(fig)
    fig = list()
    try:
        fig = ica.plot_properties(
            ecg_epochs, picks=ecg_inds, image_args={"sigma": 1.0}, show=False
        )
        for i, figure in enumerate(fig):
            # report.add_figs_to_section(
            #    figure, captions="Detected component " + str(i), section="ICA - ECG"
            # )
            close(figure)
    except:
        print("No component to remove")

    ## FIND EOG COMPONENTS
    eog_threshold = 4
    eog_epochs = create_eog_epochs(raw_data, ch_name="EEG057")
    eog_inds, eog_scores = ica.find_bads_eog(
        eog_epochs, ch_name="EEG057", threshold=eog_threshold
    )
    # TODO : if eog_inds == [] then eog_inds = [index(max(abs(eog_scores)))]
    fig = ica.plot_scores(eog_scores, eog_inds, show=False)
    # report.add_figs_to_section(
    #    fig, captions="Correlation with EOG (EEG057)", section="ICA - EOG"
    # )
    close(fig)
    fig = list()
    try:
        fig = ica.plot_properties(
            eog_epochs, picks=eog_inds, image_args={"sigma": 1.0}, show=False
        )
        for i, figure in enumerate(fig):
            report.add_figs_to_section(
                figure, captions="Detected component " + str(i), section="ICA - EOG"
            )
            close(figure)
    except:
        print("No component to remove")

    ## EXCLUDE COMPONENTS
    ica.exclude = ecg_inds
    ica.apply(raw_data)
    ica.exclude = eog_inds
    ica.apply(raw_data)
    fig = raw_data.plot(show=False)
    # Plot the clean signal.
    # report.add_figs_to_section(
    #   fig, captions="After filtering + ICA", section="Raw data"
    # )
    close(fig)
    ## SAVE PREPROCESSED FILE
    report.save(reportpath, open_browser=False, overwrite=True)
    raw_data.save(preprocpath, overwrite=True)
    del ica
    del report
    del raw_data
    del fig


##Ne pas inclure 'RS' si vous ne voulez pas faire le processing des blocs Resting State
RUN_LIST = {"pareidolia": ["1", "2", "3", "4", "5", "6", "7", "8"], "RS": ["1"]}
##Choisir le ou les participants à processer.
SUBJ_LIST = ["03", "07", "08", "09"]
SUBJ_LIST = ["03", "07", "08", "09"]
TASK_LIST = ["RS"]

# for subj, ecg, eog in zip(SUBJ_LIST, ECG_DICT, EOG_DICT):
for subj in SUBJ_LIST:
    for task in TASK_LIST:
        for run in RUN_LIST[task]:
            filename, filepath = get_pareidolia_bids(
                FOLDERPATH, subj, task, run, stage="meg", cond=None
            )
            preproc_name, preproc_path = get_pareidolia_bids(
                FOLDERPATH, subj, task, run, stage="preproc", cond=None
            )
            report_name, report_path = get_pareidolia_bids(
                FOLDERPATH, subj, task, run, stage="report_preproc", cond=None
            )

            pareidolia_preproc(filepath, preproc_path, report_path)
