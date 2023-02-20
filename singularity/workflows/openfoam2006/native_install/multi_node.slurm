#!/bin/bash

#SBATCH --job-name=openfoam-container
#SBATCH --time=02:00:00
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=3

module swap PrgEnv-cray PrgEnv-gnu
module load openfoam/2006
source $OPENFOAM_HOME/etc/bashrc


# Alternative decomposeParDict name:
decompDict="-decomposeParDict system/decomposeParDict.6"

cp -r $FOAM_TUTORIALS/incompressible/simpleFoam/motorBike .
cd motorBike
./Allclean
rm *.log

# copy motorbike surface from resources directory
cp $FOAM_TUTORIALS/resources/geometry/motorBike.obj.gz constant/triSurface/

srun -n 1 --hint=nomultithread surfaceFeatureExtract >> surfaceFeatureExtract.log
srun -n 1 --hint=nomultithread blockMesh >> blockMesh.log
srun -n 1 --hint=nomultithread decomposePar $decompDict >> decomposePar.log

# Using distributedTriSurfaceMesh?
output=$(srun -n 1 --hint=nomultithread foamDictionary -entry geometry -value system/snappyHexMeshDict | grep -q distributedTriSurfaceMesh)
if [ ! -z "$output" ]
  then
    echo "surfaceRedistributePar does not need to be run anymore"
    echo " - distributedTriSurfaceMesh will do on-the-fly redistribution"
fi

srun -n $SLURM_NTASKS -N $SLURM_NNODES --hint=nomultithread snappyHexMesh $decompDict -overwrite -parallel >> snappyHexMesh.log
srun -n $SLURM_NTASKS -N $SLURM_NNODES --hint=nomultithread topoSet $decompDict -parallel >> topoSet.log

#- For non-parallel running: - set the initial fields

#- For parallel running: set the initial fields
srun -n 1 --hint=nomultithread /project/k01/shaima0d/singularity_test/openfoam/of2006/container/restore.sh

srun -n $SLURM_NTASKS -N $SLURM_NNODES --hint=nomultithread patchSummary $decompDict -parallel >> patchSummary.log
srun -n $SLURM_NTASKS -N $SLURM_NNODES --hint=nomultithread potentialFoam $decompDict -writephi -parallel >> potentialFoam.log
srun -n $SLURM_NTASKS -N $SLURM_NNODES --hint=nomultithread checkMesh $decompDict -writeFields '(nonOrthoAngle)' -constant -parallel >> checkMesh.log

srun -n $SLURM_NTASKS -N $SLURM_NNODES --hint=nomultithread simpleFoam $decompDict -parallel >> simpleFoam.log

srun -n 1 reconstructParMesh -constant >> reconstructParMesh.log
srun -n 1 reconstructPar -latestTime >> reconstructPar.log
