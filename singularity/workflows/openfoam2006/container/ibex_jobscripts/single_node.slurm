#!/bin/bash

#SBATCH --job-name=openfoam-container
#SBATCH --ntasks=6
#SBATCH --time=02:00:00

module load openmpi/4.0.3
module load singularity
module list


export OMPI_MCA_btl=openib
export OMPI_MCA_btl_openib_allow_ib=1
export IMAGE=/ibex/scratch/shaima0d/scratch/singularity_mpi_testing/images/openfoam_openmpi_base403_2006.sif

# Alternative decomposeParDict name:
decompDict="-decomposeParDict system/decomposeParDict.6"
## Standard decomposeParDict name:
# unset decompDict
cd motorBike
singularity exec  $IMAGE ./Allclean
rm *.log

# copy motorbike surface from resources directory
singularity exec  $IMAGE cp ../../motorBike.obj.gz constant/triSurface/

mpirun -np 1 singularity exec  $IMAGE surfaceFeatureExtract >> surfaceFeatureExtract.log
mpirun -np 1 singularity exec  $IMAGE blockMesh >> blockMesh.log
mpirun -np 1 singularity exec  $IMAGE decomposePar $decompDict >> decomposePar.log

# Using distributedTriSurfaceMesh?
output=$(mpirun -np 1 singularity exec  $IMAGE foamDictionary -entry geometry -value system/snappyHexMeshDict | grep -q distributedTriSurfaceMesh)
if [ ! -z "$output" ]
  then
    echo "surfaceRedistributePar does not need to be run anymore"
    echo " - distributedTriSurfaceMesh will do on-the-fly redistribution"
fi

mpirun -np 6 singularity exec  $IMAGE snappyHexMesh $decompDict -overwrite -parallel >> snappyHexMesh.log
mpirun -np 6 singularity exec  $IMAGE topoSet $decompDict -parallel >> topoSet.log

#- For non-parallel running: - set the initial fields

#- For parallel running: set the initial fields
mpirun -np 1 singularity exec  $IMAGE ../../restore.sh

mpirun -np 6 singularity exec  $IMAGE patchSummary $decompDict -parallel >> patchSummary.log
mpirun -np 6 singularity exec  $IMAGE potentialFoam $decompDict -writephi -parallel >> potentialFoam.log
mpirun -np 6 singularity exec  $IMAGE checkMesh $decompDict -writeFields '(nonOrthoAngle)' -constant -parallel >> checkMesh.log

mpirun -np 6 singularity exec  $IMAGE simpleFoam $decompDict -parallel >> simpleFoam.log

mpirun -np 1 singularity exec  $IMAGE reconstructParMesh -constant >> reconstructParMesh.log
mpirun -np 1 singularity exec  $IMAGE reconstructPar -latestTime >> reconstructPar.log
