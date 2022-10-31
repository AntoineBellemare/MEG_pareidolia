import mne_bids
import mne
import os
import os.path as op
import numpy as np
from MEG_pareidolia_utils import *
from mne_bids import write_raw_bids, BIDSPath
import PARAMS
from PARAMS import *


EVENT_ID = {'RS': 2, 'Cross': 3,
            'Image_ON': 4, 'Image_OFF': 5, 'Resp': 6}
            #'Resp_late': 66}
# check if BIDS_PATH exists, if not, create it
if not os.path.isdir(BIDS_PATH):
    os.mkdir(BIDS_PATH)
    print('BIDS folder created at : {}'.format(BIDS_PATH))
else:
    print('{} already exists.'.format(BIDS_PATH))

# list folders in acquisition folder
recording_folders = os.listdir(ACQ_PATH)
session = 'recording'
# loop across recording folders (folder containing the recordings of the day)
for rec_date in recording_folders: # folders are named by date in format YYYYMMDD
    nfiles = 0
    filelist = os.listdir(op.join(ACQ_PATH, rec_date))
    for f in filelist:
        if 'PAREMEG' in f and '.ds' in f: 
            nfiles +=1
    if nfiles < 10:
        nfiles_str = '0'+str(nfiles)
    else:
        nfiles_str = str(nfiles)
    subjects_in_folder = np.unique([filename[1:3] for filename in filelist if 'PAREMEG' in filename ])
    for file in filelist: 
        # Create emptyroom BIDS if doesn't exist already
        if 'NOISE_noise' in file:
            for sub in subjects_in_folder:
                noise_bidspath = BIDSPath(subject=sub, session='NOISE', suffix='meg', extension='.ds', root=BIDS_PATH)
                if not op.isdir(noise_bidspath):
                    er_raw_fname = op.join(ACQ_PATH, rec_date, file)
                    er_raw = mne.io.read_raw_ctf(er_raw_fname)
                    write_raw_bids(er_raw, noise_bidspath, overwrite=True)
          
        
        #Identify Resting state blocs (first and last blocs)
        if 'PAREMEG' in file and '.ds' in file: 
            subject = file[1:3]
            if file[-5:-3] == '01': #First bloc is RS
                task = 'RS'
                run = '1'
            
            
            if file[-5:-3] == nfiles_str:
                task = 'RS'
                run = '2'


            #Identify Experimental blocs
            if file[-5:-3] != '01' and file[-5:-3] != nfiles_str:
                task = 'pareidolia'
                run = str(int(file[-5:-3])-1)

            # Rewrite in BIDS format if doesn't exist yet
            bids_basename = 'sub-{}_ses-{}_task-{}_run-{}'.format(subject, session, task, run)
            bidspath = BIDSPath(subject=subject, session=session, task=task, run=run, suffix='meg', extension='.ds', root=BIDS_PATH, datatype = 'meg')
            if not op.isdir(bidspath):
                raw_fname = op.join(ACQ_PATH, rec_date, file)
                raw = mne.io.read_raw_ctf(raw_fname, preload=False)
                if task == 'pareidolia':
                    events = mne.find_events(raw, shortest_event = 1)
                    write_raw_bids(raw, bidspath, events_data=events, event_id=EVENT_ID, overwrite=True)
                else:
                    write_raw_bids(raw, bidspath, overwrite=True)

                    #events = reformat_events(events, FDlist = None, RT_thresh = RT_thresh, task = task, run = run, slowVSfast = False, FD = 
