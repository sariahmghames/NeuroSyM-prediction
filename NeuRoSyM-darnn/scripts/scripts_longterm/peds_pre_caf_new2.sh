#!/bin/bash
#file peds_pre_caf_new2.sh
#SBATCH --job-name=run
#SBATCH -wLCH04
#SBATCH -N 1
#SBATCH --output=train_cafe_alphacnd_Nin1out_80ts_v2.out


## comments are writter like ##
## "#SBATCH" is directive for slurm controller
## use srun <command>

srun python neurosym_train.py --config1="../conf/JackRabbot_x2_longterm.json" --config2="../conf/JackRabbot_x2_longterm.json"

