#!/bin/bash 

#SBATCH -N 2
#SBATCH -n 2
#SBATCH -t 00:05:00

module swap PrgEnv-cray PrgEnv-gnu
module load cray-mpich-abi
module load singularity
module list 


export IMAGE=/project/k01/shaima0d/singularity_test/images/osu_mpich332_563.sif

export BIND="-B /opt,/usr/lib64,/etc,/sw"
export EXE=/usr/local/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_latency
echo "On same node"
srun -n 2 -N 1 hostname
srun -n 2 -N 1 --mpi=pmi2 singularity exec ${BIND} ${IMAGE} ${EXE}

echo "On two nodes"
srun -n 2 -N 2 hostname
srun -n 2 -N 2 --mpi=pmi2 singularity exec ${BIND} ${IMAGE} ${EXE}
