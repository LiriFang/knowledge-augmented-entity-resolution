#!/bin/bash
#SBATCH --account=bces-delta-gpu
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
#SBATCH --output="iTunes.EL.%j.%N.out"
##SBATCH --error="a.out.%j.%N.err"
##SBATCH --mail-user=yirenl2@illinois.edu
##SBATCH --mail-type="BEGIN,END" See sbatch or srun man pages for more email options 
 
export PYTHONPATH=$PYTHONPATH:/projects/bces/yirenl2/ReFinED/src

# python train_ditto.py --dk entityLinking --task Dirty/DBLP-GoogleScholar
# python train_ditto.py --dk entityLinking --task Textual/Abt-Buy
# python train_ditto.py --dk entityLinking --task Dirty/iTunes-Amazon

# python train_ditto.py --dk sherlock --task Dirty/iTunes-Amazon --batch_size 8 --max_len 512
# python train_ditto.py --dk sherlock --task Textual/Abt-Buy --batch_size 8 --max_len 512
python train_ditto.py --dk sherlock --task Structured/iTunes-Amazon
# python train_ditto.py --dk sherlock --task Textual/Abt-Buy --batch_size 8 --max_len 512
