#!/bin/bash

#SBATCH --job-name=openfoam-container
#SBATCH --time=02:00:00
#SBATCH --nodes=1

module swap PrgEnv-cray PrgEnv-gnu
module use /project/k01/shaima0d/software/cle7up01/modulefiles
module load openmpi/4.0.3
module load singularity
module list

export UCX_NET_DEVICES=ipogif0

export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
export SINGULARITYENV_LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH:/opt/cray/wlm_detect/default/lib64:/etc/alternatives:/app/ThirdParty-v2006/platforms/linux64Gcc/fftw-3.3.7/lib64:/app/ThirdParty-v2006/platforms/linux64Gcc/CGAL-4.12.2/lib64:/app/ThirdParty-v2006/platforms/linux64Gcc/boost_1_66_0/lib64:/app/OpenFOAM-v2006/platforms/linux64GccDPInt64Opt/lib/openmpi-system:/app/ThirdParty-v2006/platforms/linux64GccDPInt64/lib/openmpi-system:/usr/local/lib:/root/OpenFOAM-v2006/platforms/linux64GccDPInt64Opt/lib:/app/OpenFOAM-v2006/site/2006/platforms/linux64GccDPInt64Opt/lib:/app/OpenFOAM-v2006/platforms/linux64GccDPInt64Opt/lib:/app/ThirdParty-v2006/platforms/linux64GccDPInt64/lib:/app/OpenFOAM-v2006/platforms/linux64GccDPInt64Opt/lib/dummy:/usr/lib/x86_64-linux-gnu:/usr/lib64:/usr/lib
export SINGULARITYENV_APPEND_PATH=$PATH:/root/OpenFOAM-v2006/platforms/linux64GccDPInt64Opt/bin:/app/OpenFOAM-v2006/site/2006/platforms/linux64GccDPInt64Opt/bin:/app/OpenFOAM-v2006/platforms/linux64GccDPInt64Opt/bin:/app/OpenFOAM-v2006/bin:/app/OpenFOAM-v2006/wmake:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin

export IMAGE=/scratch/$USER/images/openfoam_openmpi_base403_2006.sif
export BIND_MOUNT="-B $UCX_DIR,$OPENMPI_DIR,/usr/lib64,/opt,/etc"

# Alternative decomposeParDict name:
decompDict="-decomposeParDict system/decomposeParDict.6"
## Standard decomposeParDict name:
# unset decompDict
cp /sw/xc40cle7up03/openfoam/2006/cle7_gnu8.3.0/OpenFOAM-v2006/tutorials/resources/geometry/motorBike.obj.gz .                      
cp -r /sw/xc40cle7up03/openfoam/2006/cle7_gnu8.3.0/OpenFOAM-v2006/tutorials/incompressible/simpleFoam/motorBike .
cd motorBike
singularity exec $BIND_MOUNT $IMAGE ./Allclean
rm *.log

# copy motorbike surface from resources directory
singularity exec $BIND_MOUNT $IMAGE cp $PWD/../motorBike.obj.gz constant/triSurface/

mpirun -np 1 singularity exec $BIND_MOUNT $IMAGE surfaceFeatureExtract >> surfaceFeatureExtract.log
mpirun -np 1 singularity exec $BIND_MOUNT $IMAGE blockMesh >> blockMesh.log
mpirun -np 1 singularity exec $BIND_MOUNT $IMAGE decomposePar $decompDict >> decomposePar.log

# Using distributedTriSurfaceMesh?
output=$(mpirun -np 1 singularity exec $BIND_MOUNT $IMAGE foamDictionary -entry geometry -value system/snappyHexMeshDict | grep -q distributedTriSurfaceMesh)
if [ ! -z "$output" ]
  then
    echo "surfaceRedistributePar does not need to be run anymore"
    echo " - distributedTriSurfaceMesh will do on-the-fly redistribution"
fi

mpirun -np 6 singularity exec $BIND_MOUNT $IMAGE snappyHexMesh $decompDict -overwrite -parallel >> snappyHexMesh.log
mpirun -np 6 singularity exec $BIND_MOUNT $IMAGE topoSet $decompDict -parallel >> topoSet.log

#- For non-parallel running: - set the initial fields

#- For parallel running: set the initial fields
mpirun -np 1 singularity exec $BIND_MOUNT $IMAGE $PWD/../restore.sh 

mpirun -np 6 singularity exec $BIND_MOUNT $IMAGE patchSummary $decompDict -parallel >> patchSummary.log
mpirun -np 6 singularity exec $BIND_MOUNT $IMAGE potentialFoam $decompDict -writephi -parallel >> potentialFoam.log
mpirun -np 6 singularity exec $BIND_MOUNT $IMAGE checkMesh $decompDict -writeFields '(nonOrthoAngle)' -constant -parallel >> checkMesh.log

mpirun -np 6 singularity exec $BIND_MOUNT $IMAGE simpleFoam $decompDict -parallel >> simpleFoam.log

mpirun -np 1 singularity exec $BIND_MOUNT $IMAGE reconstructParMesh -constant >> reconstructParMesh.log
mpirun -np 1 singularity exec $BIND_MOUNT $IMAGE reconstructPar -latestTime >> reconstructPar.log
