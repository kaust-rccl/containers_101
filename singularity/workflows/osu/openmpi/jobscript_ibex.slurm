#!/bin/bash 

#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:v100:2
#SBATCH --time=00:05:00
#SBATCH --account=ibex-cs

module load singularity
module load openmpi/4.0.3-cuda10.2
module list

export OMPI_MCA_btl=openib
export OMPI_MCA_btl_openib_allow_ib=1
export IMAGE=/ibex/scratch/shaima0d/scratch/singularity_mpi_testing/images/osu_cuda_openmpi403_563.sif

export EXE_lat=/usr/local/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_latency
export EXE_bw=/usr/local/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw


echo "On same node"
mpirun -n 2 --map-by ppr:2:node hostname
mpirun -n 2 --map-by ppr:2:node singularity exec --nv ${IMAGE} ${EXE_lat} D D
mpirun -n 2 --map-by ppr:2:node singularity exec --nv ${IMAGE} ${EXE_bw} D D



echo "On two nodes"
mpirun -n 2 --map-by ppr:1:node hostname
mpirun -n 2 --map-by ppr:1:node singularity exec --nv ${IMAGE} ${EXE_lat} D D
mpirun -n 2 --map-by ppr:1:node singularity exec --nv ${IMAGE} ${EXE_bw} D D

