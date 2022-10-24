#!/bin/bash
#SBATCH --account=bbno-delta-gpu
### GPU options ###
#SBATCH --partition=gpuA100x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
##SBATCH --gpu-bind=none     # <- or closest
#SBATCH --tasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS
#SBATCH --mem=40g
#SBATCH --job-name=ditto-sherlock
#SBATCH --time=01:00:00      # hh:mm:ss for the job
#SBATCH --output="Exp_Logging/DBLP.GoogleScholar.sherlock.structured.wt.prep.%j.%N.out"
##SBATCH --error="a.out.%j.%N.err"
##SBATCH --mail-user=yirenl2@illinois.edu
##SBATCH --mail-type="BEGIN,END" See sbatch or srun man pages for more email options 
 
export PYTHONPATH=$PYTHONPATH:/projects/bbno/yirenl2/ReFinED/src

# python train_ditto.py --dk entityLinking --task Dirty/DBLP-GoogleScholar
# python train_ditto.py --dk entityLinking --task Textual/Abt-Buy
# python train_ditto.py --dk entityLinking --task Dirty/iTunes-Amazon
# python train_ditto.py --dk sherlock --task Dirty/iTunes-Amazon
# python train_ditto.py --dk sherlock --task Structured/DBLP-GoogleScholar --kbert True
python train_ditto.py --dk sherlock --task Dirty/DBLP-GoogleScholar --kbert True