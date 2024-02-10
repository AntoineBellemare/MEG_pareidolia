#!/bin/sh
#SBATCH --account=def-kjerbi
#SBATCH --mem=128G
#SBATCH --job-name=FOOOF_pareidolia
#SBATCH --time=5:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24
$HOME/projects/def-kjerbi/abel/MEG_pareidolia/bin/python $HOME/projects/def-kjerbi/abel/MEG_pareidolia/FOOOF_cc.py -s $1 -r $2