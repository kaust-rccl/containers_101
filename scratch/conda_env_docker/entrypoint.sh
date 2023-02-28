#!/bin/bash
#Initialize conda
source /software/miniconda3/etc/profile.d/conda.sh
conda activate myenv 
exec "$@"
