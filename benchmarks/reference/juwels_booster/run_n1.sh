#!/bin/bash
srun -n4 ./dslash.py --mpi 4.1.1.1 --grid 32.32.32.64
