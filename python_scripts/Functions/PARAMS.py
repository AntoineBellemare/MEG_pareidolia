FOLDERPATH = r'D:\PsychoPy_MEG\BIDS_data'
BIDS_PATH = r'D:\PsychoPy_MEG\BIDS_data'
ACQ_PATH = r'D:\PsychoPy_MEG\acquisitions'

FREQ_BANDS = [[1, 3], [4, 7], [8, 12], [13, 19], [20, 29], [30, 45], [46, 60]]
FREQ_NAMES = ['delta', 'theta', 'alpha', 'low-beta', 'high-beta', 'gamma1', 'gamma2']

RT_thresh = 4000

MEG_atlas = {'CL': list(range(0, 24)), 'FL': list(range(24, 57)), 'OL': list(range(57, 76)), 
             'PL': list(range(76, 97)), 'TL': list(range(97, 131)),
             'CR': list(range(131, 153)), 'FR': list(range(153, 186)), 'OR': list(range(186, 204)), 
             'PR': list(range(204, 226)), 'TR': list(range(226, 259)), 'CZ': list(range(259, 263)),
             'Fz': list(range(263, 266)), 'OZ': list(range(266, 269)), 'PZ': list(range(269, 270))
            }