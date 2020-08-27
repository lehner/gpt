module load gcc/6.4.0
module load cuda/10.1.168
module load python/3.7.0
export LD_LIBRARY_PATH=/ccs/home/lehner/gpt/dependencies/mpfr-4.0.2/build/lib:$LD_LIBRARY_PATH
export shm=512
export OMP_NUM_THREADS=6
export GRID_ALLOC_NCACHE_LARGE=16
export GRID_ALLOC_NCACHE_SMALL=32

source ~/gpt/scripts/source.sh

SCRIPT=$@

jsrun --smpiargs="-gpu" --nrs 6 --rs_per_host 6  --tasks_per_rs 1 --cpu_per_rs 6 --gpu_per_rs 1 ${SCRIPT} --mpi 6.1.1.1 --comms-overlap \
 --comms-concurrent --shm $shm --decomposition \
 --device-mem 4000 2>&1

