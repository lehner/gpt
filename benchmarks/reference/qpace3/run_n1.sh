#!/bin/bash
#SBATCH --partition=qp48
#SBATCH --time=30:00
#SBATCH --nodes=1
#SBATCH --account=ur
srun hostname
source ~/gpt/scripts/source.sh

export I_MPI_FABRICS=shm:ofi
export I_MPI_FALLBACK=0
export I_MPI_DEBUG=5

cat /proc/cpuinfo

OMP_NUM_THREADS=64 srun  \
./dslash.py --decomposition --dslash-asm --grid 32.32.32.32 --mpi 1.1.1.1



