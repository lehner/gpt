#!/bin/bash
srun -N1 -n4 benchmarks/dslash.py --shm 2048 --device-mem 16000 --grid 64.32.32.32 --mpi 4.1.1.1 --accelerator-threads 8 --Ls 12
