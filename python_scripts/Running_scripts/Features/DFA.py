import multiprocessing
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy.io as sio
import sys
from scipy.signal import hilbert
import h5py
import neurokit2 as nk
import pandas as pd

sys.path.insert(0, "C:/Users/Antoine/github/MEG_pareidolia/python_scripts/Functions")
from MEG_pareidolia_utils import *
import PARAMS
from PARAMS import *

# Ensure the additional scripts are available


# Define the worker function for parallel processing
def process_trial(
    subj, task, run, trial, hilbert_data, min_freq, max_freq, progress_counter
):
    amplitude_envelope = hilbert_data[trial]
    dfa_results = []
    for electrode_index, envelope in enumerate(amplitude_envelope):
        envelope = rescale_array(envelope, -1, 1)
        dfa_exponent, info = nk.fractal_dfa(envelope)
        print("DFA exponent", dfa_exponent)
        dfa_results.append(
            {
                "Subject": subj,
                "Task": task,
                "Run": run,
                "Trial": trial,
                "Frequency_Band": f"{min_freq}-{max_freq}",
                "Electrode": electrode_index,
                "DFA_Exponent": dfa_exponent,
            }
        )
    progress_counter.value += 1
    return dfa_results


def rescale_array(arr, new_min, new_max):
    min_arr = np.min(arr)
    max_arr = np.max(arr)
    scaled_array = new_min + (
        (arr - min_arr) * (new_max - new_min) / (max_arr - min_arr)
    )
    return scaled_array


def main():
    # Define subject, run, and task lists
    RUN_LIST = {
        "pareidolia": ["1", "2", "3", "4", "5", "6", "7", "8"],
        "RS": ["1", "2"],
    }
    SUBJ_LIST = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]
    TASK_LIST = ["pareidolia"]

    # Create a pool of worker processes
    manager = multiprocessing.Manager()
    progress_counter = manager.Value("i", 0)

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() // 4)

    for subj in SUBJ_LIST:
        for task in TASK_LIST:
            for run in RUN_LIST[task]:
                print("Run", run)
                # Load Hilbert data
                # Load the Hilbert transform data
                hilbert_file, hilbert_path = get_pareidolia_bids(
                    FOLDERPATH, subj, task, run, stage="Hilbert_long"
                )
                with h5py.File(hilbert_path, "r") as f:
                    hilbert_data = f["hilbert_data"][:]
                total_tasks = sum(
                    len(RUN_LIST[task]) * len(FREQ_BANDS) * hilbert_data.shape[2]
                    for task in TASK_LIST
                )
                results = []
                for i, (min_freq, max_freq) in enumerate(FREQ_BANDS):
                    for trial in range(hilbert_data.shape[2]):
                        result = pool.apply_async(
                            process_trial,
                            (
                                subj,
                                task,
                                run,
                                trial,
                                hilbert_data[i, 0, :],
                                min_freq,
                                max_freq,
                                progress_counter,
                            ),
                        )
                        results.append(result)

                for r in results:
                    r.get()
                    print(
                        f"Progress: {progress_counter.value}/{total_tasks} tasks completed"
                    )

                # Gather and flatten results
                dfa_results = [
                    item for sublist in [r.get() for r in results] for item in sublist
                ]

                # Convert results to DataFrame and save to CSV
                dfa_df = pd.DataFrame(dfa_results)
                dfa_file, dfa_path = get_pareidolia_bids(
                    FOLDERPATH, subj, task, run, stage="DFA"
                )
                dfa_df.to_csv(dfa_path, index=False)

    # Close the pool and wait for all processes to complete
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
