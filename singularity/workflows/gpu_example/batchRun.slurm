#!/bin/bash 

#SBATCH --gpus=2
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=2
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00

module load singularity
module list

export IMAGE=/ibex/ai/home/shaima0d/singularity_images/ngc/pytorch_22.12-py3.sif


export SINGULARITYENV_OMP_NUM_THREADS=1

export SINGULARITYENV_NCCL_DEBUG=INFO
export SINGULARITYENV_NCCL_SOCKET_IFNAME=ib0
export SINGULARITYENV_PYTHONFAULTHANDLER=1


# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
echo "Node IDs of participating nodes ${nodes_array[*]}"

# Get the IP address and set port for MASTER node
head_node="${nodes_array[0]}"
echo "Getting the IP address of the head node ${head_node}"
master_ip=$(srun -n 1 -N 1 --gpus=1 -w ${head_node} /bin/hostname -I | cut -d " " -f 2)
master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "head node is ${master_ip}:${master_port}"


for (( i=0; i< ${SLURM_NNODES}; i++ ))
do
     srun -n 1 -N 1 -c ${SLURM_CPUS_PER_TASK} -w ${nodes_array[i]} --gpus=${SLURM_GPUS_PER_NODE}  \
      singularity exec --nv -B /ibex/ai/reference:/reference  ${IMAGE} \
      python -m torch.distributed.launch --use_env \
     --nproc_per_node=${SLURM_GPUS_PER_NODE} --nnodes=${SLURM_NNODES} --node_rank=${i} \
     --master_addr=${master_ip} --master_port=${master_port} \
     ddp.py --epochs=2 --lr=0.001 --num-workers=${SLURM_CPUS_PER_TASK} --batch-size=256 &
done
wait

