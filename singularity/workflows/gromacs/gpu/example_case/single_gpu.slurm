#!/bin/bash

#SBATCH --job-name="gromacs"
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time=1:00:00
#----------------------------------------------------------#

module load singularity
module load openmpi/4.0.3-cuda11.0
module list
export OMPI_MCA_btl=openib
export OMPI_MCA_btl_openib_allow_ib=1

export SINGULARITYENV_LD_LIBRARY_PATH=/usr/local/gromacs/sm70/lib:/usr/local/fftw/lib:/usr/local/cuda/lib:/usr/local/cuda/lib64:/.singularity.d/libs
export SINGULARITYENV_GMXBIN=/usr/local/gromacs/sm70/bin
export SINGULARITYENV_CUDA_HOME=/sw/csgv/cuda/11.0.1
export SINGULARITYENV_GMXDATA=/usr/local/gromacs/sm70/share/gromacs
export SINGULARITYENV_GMXLDLIB=/usr/local/gromacs/sm70/lib
export SINGULARITYENV_GMXMAN=/usr/local/gromacs/sm70/share/man
export SINGULARITYENV_PATH=/usr/local/gromacs/sm70/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export SINGULARITYENV_GROMACS_DIR=/usr/local/gromacs/sm70

export IMAGE=gromacs_2020.2.sif
export GROMACS_HOME=/usr/local/gromacs/sm70


cp ../example_case/* .

# Step One: Prepare the Topology, Step Two: Examine the Topology
grep -v HOH 1AKI.pdb > 1AKI_clean.pdb
echo 15 | mpirun -np 1 singularity exec --nv ${IMAGE} ${GROMACS_HOME}/bin/gmx pdb2gmx -f 1AKI_clean.pdb -o 1AKI_processed.gro -water spce

# Step Three: Defining the Unit Cell & Adding Solvent
mpirun -np 1 singularity exec --nv ${IMAGE} ${GROMACS_HOME}/bin/gmx editconf -f 1AKI_processed.gro -o 1AKI_newbox.gro -c -d 1.0 -bt cubic
mpirun -np 1 singularity exec --nv ${IMAGE} ${GROMACS_HOME}/bin/gmx solvate -cp 1AKI_newbox.gro -cs spc216.gro -o 1AKI_solv.gro -p topol.top

# Step Four: Adding Ions
mpirun -np 1 singularity exec --nv ${IMAGE} ${GROMACS_HOME}/bin/gmx grompp -f ions.mdp -c 1AKI_solv.gro -p topol.top -o ions.tpr
echo 13 | mpirun -np 1 singularity exec --nv ${IMAGE} ${GROMACS_HOME}/bin/gmx genion -s ions.tpr -o 1AKI_solv_ions.gro -p topol.top -pname NA -nname CL -neutral

# Step Five: Energy Minimization
mpirun -np 1 singularity exec --nv ${IMAGE} ${GROMACS_HOME}/bin/gmx grompp -f minim.mdp -c 1AKI_solv_ions.gro -p topol.top -o em.tpr
mpirun -np 1 singularity exec --nv ${IMAGE} ${GROMACS_HOME}/bin/gmx mdrun -v -deffnm em
echo 10 0 | mpirun -np 1 singularity exec --nv ${IMAGE} ${GROMACS_HOME}/bin/gmx energy -f em.edr -o potential.xvg

# Step Six: Equilibration
mpirun -np 1 singularity exec --nv ${IMAGE} ${GROMACS_HOME}/bin/gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr
mpirun -np 1 -N 1  singularity exec --nv ${IMAGE} ${GROMACS_HOME}/bin/gmx mdrun -deffnm nvt
echo 16 0 | mpirun -np 1 singularity exec --nv ${IMAGE} ${GROMACS_HOME}/bin/gmx energy -f nvt.edr -o temperature.xvg

# Step Seven: Equilibration, Part 2
mpirun -np 1 singularity exec --nv ${IMAGE} ${GROMACS_HOME}/bin/gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr
mpirun -np 1 -N 1  singularity exec --nv ${IMAGE} ${GROMACS_HOME}/bin/gmx mdrun -deffnm npt
echo 18 0 | mpirun -np 1 singularity exec --nv ${IMAGE} ${GROMACS_HOME}/bin/gmx energy -f npt.edr -o pressure.xvg
echo 24 0 | mpirun -np 1 singularity exec --nv ${IMAGE} ${GROMACS_HOME}/bin/gmx energy -f npt.edr -o density.xvg

# Step Eight: Production MD
mpirun -np 1 singularity exec --nv ${IMAGE} ${GROMACS_HOME}/bin/gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md_0_1.tpr
mpirun -np 1 -N 1 singularity exec --nv ${IMAGE} ${GROMACS_HOME}/bin/gmx mdrun -deffnm md_0_1

# Step Nine: Analysis
echo 1 0 | mpirun -np 1 singularity exec --nv ${IMAGE} ${GROMACS_HOME}/bin/gmx trjconv -s md_0_1.tpr -f md_0_1.xtc -o md_0_1_noPBC.xtc -pbc mol -center
echo 4 4 | mpirun -np 1 singularity exec --nv ${IMAGE} ${GROMACS_HOME}/bin/gmx rms -s md_0_1.tpr -f md_0_1_noPBC.xtc -o rmsd.xvg -tu ns
echo 4 4 | mpirun -np 1 singularity exec --nv ${IMAGE} ${GROMACS_HOME}/bin/gmx rms -s em.tpr -f md_0_1_noPBC.xtc -o rmsd_xtal.xvg -tu ns
echo 1 | mpirun -np 1 singularity exec --nv ${IMAGE} ${GROMACS_HOME}/bin/gmx gyrate -s md_0_1.tpr -f md_0_1_noPBC.xtc -o gyrate.xvg

