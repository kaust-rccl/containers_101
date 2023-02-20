#!/bin/bash 

#SBATCH -N 1
#SBATCH -t 01:00

module swap PrgEnv-cray PrgEnv-gnu
module load cray-mpich-abi
module load singularity
module list 


#export IMAGE=/project/k01/shaima0d/singularity_test/images/hpl_mpich314_latest.sif
export IMAGE=/project/k01/shaima0d/singularity_test/images/hpl_mpich_2.3.0.sif

export BIND="-B /opt,/var,/usr/lib64,/etc,/sw"
export EXE=/hpl/bin/ubuntu/xhpl
echo "On same node"
srun -n 4 -N 1 --mpi=pmi2 singularity exec ${BIND} ${IMAGE} ${EXE}

