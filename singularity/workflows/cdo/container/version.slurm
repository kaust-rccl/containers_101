#!/bin/bash
#SBATCH --job-name=cdo-container
#SBATCH --time=00:05:00
#SBATCH --nodes=1
module load singularity
srun -N 1 singularity run /scratch/$USER/images/cdo_gnu_1.9.10.sif cdo --version
