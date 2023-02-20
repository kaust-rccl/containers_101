#!/bin/bash
#SBATCH --partition=workq
#SBATCH --job-name="gromacs"
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=8
#SBATCH --ntasks-per-socket=4
#SBATCH --time=4:00:00
#----------------------------------------------------------#
module swap PrgEnv-cray PrgEnv-gnu
module use /project/k01/shaima0d/software/cle7up01/modulefiles
module load openmpi/4.0.3
module load singularity
module list

export UCX_NET_DEVICES=ipogif0

export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
export SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cray/wlm_detect/default/lib64:/etc/alternatives:/usr/local/gromacs/lib:/usr/local:/usr/lib64:/usr/lib
export SINGULARITYENV_APPEND_PATH=$PATH

export IMAGE=/project/k01/shaima0d/singularity_test/images/gromacs_cpu_latest.sif
export BIND_MOUNT="-B $UCX_DIR,$OPENMPI_DIR,/usr/lib64,/opt"
#export SINGUALRITYENV_GROMACS_USE="" 
export SINGULARITYENV_OMP_NUM_THREADS=1
export GROMACS_HOME=/usr/local/gromacs

# Step One: Prepare the Topology, Step Two: Examine the Topology
grep -v HOH 1AKI.pdb > 1AKI_clean.pdb
echo 15 | mpirun -np 1 singularity exec ${BIND_MOUNT} ${IMAGE} ${GROMACS_HOME}/bin/gmx_mpi pdb2gmx -f 1AKI_clean.pdb -o 1AKI_processed.gro -water spce

# Step Three: Defining the Unit Cell & Adding Solvent
mpirun -np 1 singularity exec ${BIND_MOUNT} ${IMAGE} ${GROMACS_HOME}/bin/gmx_mpi editconf -f 1AKI_processed.gro -o 1AKI_newbox.gro -c -d 1.0 -bt cubic
mpirun -np 1 singularity exec ${BIND_MOUNT} ${IMAGE} ${GROMACS_HOME}/bin/gmx_mpi solvate -cp 1AKI_newbox.gro -cs spc216.gro -o 1AKI_solv.gro -p topol.top

# Step Four: Adding Ions
mpirun -np 1 singularity exec ${BIND_MOUNT} ${IMAGE} ${GROMACS_HOME}/bin/gmx_mpi grompp -f ions.mdp -c 1AKI_solv.gro -p topol.top -o ions.tpr
echo 13 | mpirun -np 1 singularity exec ${BIND_MOUNT} ${IMAGE} ${GROMACS_HOME}/bin/gmx_mpi genion -s ions.tpr -o 1AKI_solv_ions.gro -p topol.top -pname NA -nname CL -neutral

# Step Five: Energy Minimization
mpirun -np 1 singularity exec ${BIND_MOUNT} ${IMAGE} ${GROMACS_HOME}/bin/gmx_mpi grompp -f minim.mdp -c 1AKI_solv_ions.gro -p topol.top -o em.tpr
mpirun -np 1 singularity exec ${BIND_MOUNT} ${IMAGE} ${GROMACS_HOME}/bin/gmx_mpi mdrun -v -deffnm em
echo 10 0 | mpirun -np 1 singularity exec ${BIND_MOUNT} ${IMAGE} ${GROMACS_HOME}/bin/gmx_mpi energy -f em.edr -o potential.xvg

# Step Six: Equilibration
mpirun -np 1 singularity exec ${BIND_MOUNT} ${IMAGE} ${GROMACS_HOME}/bin/gmx_mpi grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr
mpirun -np 32 --map-by ppr:8:node singularity exec ${BIND_MOUNT} ${IMAGE} ${GROMACS_HOME}/bin/gmx_mpi mdrun -deffnm nvt
echo 16 0 | mpirun -np 1 singularity exec ${BIND_MOUNT} ${IMAGE} ${GROMACS_HOME}/bin/gmx_mpi energy -f nvt.edr -o temperature.xvg

# Step Seven: Equilibration, Part 2
mpirun -np 1 singularity exec ${BIND_MOUNT} ${IMAGE} ${GROMACS_HOME}/bin/gmx_mpi grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr
mpirun -np 32 --map-by ppr:8:node singularity exec ${BIND_MOUNT} ${IMAGE} ${GROMACS_HOME}/bin/gmx_mpi mdrun -deffnm npt
echo 18 0 | mpirun -np 1 singularity exec ${BIND_MOUNT} ${IMAGE} ${GROMACS_HOME}/bin/gmx_mpi energy -f npt.edr -o pressure.xvg
echo 24 0 | mpirun -np 1 singularity exec ${BIND_MOUNT} ${IMAGE} ${GROMACS_HOME}/bin/gmx_mpi energy -f npt.edr -o density.xvg

# Step Eight: Production MD
mpirun -np 1 singularity exec ${BIND_MOUNT} ${IMAGE} ${GROMACS_HOME}/bin/gmx_mpi grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md_0_1.tpr
mpirun -np 32 --map-by ppr:8:node singularity exec ${BIND_MOUNT} ${IMAGE} ${GROMACS_HOME}/bin/gmx_mpi mdrun -deffnm md_0_1

# Step Nine: Analysis
echo 1 0 | mpirun -np 1 singularity exec ${BIND_MOUNT} ${IMAGE} ${GROMACS_HOME}/bin/gmx_mpi trjconv -s md_0_1.tpr -f md_0_1.xtc -o md_0_1_noPBC.xtc -pbc mol -center
echo 4 4 | mpirun -np 1 singularity exec ${BIND_MOUNT} ${IMAGE} ${GROMACS_HOME}/bin/gmx_mpi rms -s md_0_1.tpr -f md_0_1_noPBC.xtc -o rmsd.xvg -tu ns
echo 4 4 | mpirun -np 1 singularity exec ${BIND_MOUNT} ${IMAGE} ${GROMACS_HOME}/bin/gmx_mpi rms -s em.tpr -f md_0_1_noPBC.xtc -o rmsd_xtal.xvg -tu ns
echo 1 | mpirun -np 1 singularity exec ${BIND_MOUNT} ${IMAGE} ${GROMACS_HOME}/bin/gmx_mpi gyrate -s md_0_1.tpr -f md_0_1_noPBC.xtc -o gyrate.xvg
