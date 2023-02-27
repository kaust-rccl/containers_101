#!/bin/bash 

#SBATCH -N 2
#SBATCH -n 4
#SBATCH -t 00:05:00

module swap PrgEnv-$(echo $PE_ENV | tr [:upper:] [:lower:]) PrgEnv-gnu
module use /project/k01/shaima0d/software/cle7up01/modulefiles
module load openmpi/4.0.3
module load singularity

export UCX_NET_DEVICES=ipogif0

export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
#export SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cray/wlm_detect/default/lib64:/etc/alternatives:/usr/local:/usr/lib64:/usr/lib
export SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local:/usr/lib64:/usr/lib
export SINGULARITYENV_APPEND_PATH=$PATH

export IMAGE=/scratch/$USER/images/openmpi_base_403.sif
export BIND_MOUNT="-B $UCX_DIR,$OPENMPI_DIR,/sw,/usr/lib64,/opt,/etc"
export EXE=/scratch/$USER/trainings/containers/containers_101/singularity/workflows/openmpi/hello

echo "On same node"
mpirun -n 2 -N 2 hostname
mpirun -n 2 -N 2 singularity exec ${BIND_MOUNT} ${IMAGE} ${EXE} 

echo "On two nodes"
mpirun -n 2 -N 1 hostname
mpirun -n 2 -N 1 singularity exec ${BIND_MOUNT} ${IMAGE} ${EXE} 
