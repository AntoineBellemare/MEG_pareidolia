for SUB in 00 01 02 03 04 05 06 07 08 09 10 11;
do
    for RUN in 1 2 3 4 5 6 7 8;
    do
        sbatch /home/abel/projects/def-kjerbi/abel/MEG_pareidolia/start_job.sh $SUB $RUN
    done
done