#!/bin/bash 

#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH -t 01:00:00

module load openmpi/4.0.3
module load singularity



export IMAGE=/ibex/ai/home/shaima0d/singularity_images/hpl_openmpi403_230.sif
export BIND_MOUNT="-B /sw,/usr/lib64,/opt,/etc"
export EXE=/hpl/bin/ubuntu/xhpl

echo "On same node"

mpirun -n $SLURM_NTASKS singularity exec ${BIND_MOUNT} ${IMAGE} ${EXE}


