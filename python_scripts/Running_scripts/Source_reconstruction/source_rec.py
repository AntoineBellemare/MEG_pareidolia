import mne
from mne_bids import make_bids_basename, read_raw_bids

# specify the path to the BIDS directory
bids_root = '/path/to/your/bids/directory'

# specify the subject and session information
subject = '01'
session = '01'

# create the BIDS basename
bids_basename = make_bids_basename(subject=subject, session=session, task='mytask',
                                   acquisition='meg', suffix='meg')

# read the MEG data in BIDS format
raw = read_raw_bids(bids_basename, bids_root=bids_root)

# perform coregistration using the fiducials
mne.coreg.coregistration(raw, subject=subject, subjects_dir='/path/to/freesurfer/subjects/dir')

# compute the forward solution using a template-based head model
fwd = mne.make_forward_solution(raw.info, trans=None, src='ico-4', bem='sample', meg=True, eeg=False)

# estimate the neural sources using minimum norm estimation (MNE)
inverse_operator = mne.minimum_norm.make_inverse_operator(raw.info, fwd, loose=1)
stc = mne.minimum_norm.apply_inverse(raw, inverse_operator)

# visualize the results
stc.plot(subjects_dir='/path/to/freesurfer/subjects/dir')



## FREESURFER
# recon-all -subjid my_subject_id -sd path/to/output/directory -template ICBM152