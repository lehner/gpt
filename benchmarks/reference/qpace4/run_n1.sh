#!/bin/bash
module load mpi/openmpi-aarch64
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export GPT_REPOSITORY=/home/lec17310/gpt-repository
srun $@ --mpi 1.1.1.1 --mpi 1.1.1 --dslash-asm
