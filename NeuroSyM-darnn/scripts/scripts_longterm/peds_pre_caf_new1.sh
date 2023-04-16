#!/bin/bash
#file peds_pre_caf_new1.sh
#SBATCH --job-name=run
#SBATCH -wLCH03
#SBATCH -N 1
#SBATCH --output=cafe_alphacnd_Nin1out_48ts_v1.out


## comments are writter like ##
## "#SBATCH" is directive for slurm controller
## use srun <command>


srun python neurosym_train.py --config1="../conf/JackRabbot_x1_longterm.json" --config2="../conf/JackRabbot_y1_longterm.json"

