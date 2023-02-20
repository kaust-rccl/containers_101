#!/bin/bash 

#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH -t 01:00:00

module swap PrgEnv-$(echo $PE_ENV | tr [:upper:] [:lower:]) PrgEnv-gnu
module load openmpi/4.0.3
module load singularity


export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
export SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cray/wlm_detect/default/lib64:/etc/alternatives:/usr/lib/x86_64-linux-gnu:/usr/local:/usr/lib:/usr/lib64
export SINGULARITYENV_APPEND_PATH=$PATH

export IMAGE=/project/k01/shaima0d/singularity_test/images/hpl_ompi_fpic.sif
export BIND_MOUNT="-B /sw,/usr/lib64,/opt,/etc"
export EXE=/hpl/bin/ubuntu/xhpl

echo "On same node"

#mpirun -n 4 -N 2 singularity exec ${BIND_MOUNT} ${IMAGE} ${EXE}
mpirun -n $SLURM_NTASKS singularity exec ${BIND_MOUNT} ${IMAGE} ${EXE}


