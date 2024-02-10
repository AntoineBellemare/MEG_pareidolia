FOLDERPATH = r"D:\Science\PsychoPy_MEG\BIDS_data"
BIDS_PATH = r"D:\PsychoPy_MEG\BIDS_data"
ACQ_PATH = r"D:\PsychoPy_MEG\acquisitions"

FREQ_BANDS = [[1, 3], [3, 7], [7, 12], [12, 20], [20, 30], [30, 45], [45, 60]]
FREQ_BANDS2 = [
    [1, 3],
    [3, 7],
    [7, 12],
    [12, 20],
    [20, 30],
    [30, 45],
    [45, 60],
    [60, 90],
]
FREQ_BANDS_nodelta = [
    [3, 7],
    [7, 12],
    [12, 20],
    [20, 30],
    [30, 45],
    [45, 60],
]

FREQ_BANDS3 = [
    [2, 3],
    [3, 7],
    [7, 12],
    [12, 20],
    [20, 30],
    [30, 45],
    [45, 60],
]
FREQ_BANDS4 = [
    [2, 4],
    [4, 8],
    [8, 12],
    [12, 20],
    [20, 30],
    [30, 45],
    [45, 60],
    [60, 90],
]
FREQ_BANDS_multigamma = [
    [2, 4],
    [4, 8],
    [8, 12],
    [12, 20],
    [20, 30],
    [30, 60],
    [60, 90],
    [90, 120],
]

FREQ_NAMES2 = [
    "delta",
    "theta",
    "alpha",
    "low-beta",
    "high-beta",
    "gamma1",
    "gamma2",
    "gamma3",
]
FREQ_NAMES = ["delta", "theta", "alpha", "low-beta", "high-beta", "gamma1", "gamma2"]

RT_thresh = 4000

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
