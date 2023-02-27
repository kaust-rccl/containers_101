#!/bin/bash

#SBATCH -n 1 
#SBATCH -t 00:10:00 
#SBATCH -A ibex-cs

module load singularity

singularity build --fakeroot --force ./conda_test.modified ./singularity_entrypoint.def
