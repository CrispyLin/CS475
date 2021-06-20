#!/bin/bash
#SBATCH -J montecarlo
#SBATCH -A CS475-575
#SBATCH -p class
#SBATCH --gres=gpu:1
#SBATCH -o montecarlo.out
#SBATCH -e montecarlo.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=linxinw@oregonstate.edu

rm -f montecarlo.csv

printf ",16,32,64,128\n" >> montecarlo.csv
for t in 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576
do
    printf "$t," >> montecarlo.csv
    for bz in 16 32 64 128
    do  
        /usr/local/apps/cuda/cuda-10.1/bin/nvcc -DTHREADS_PER_BLOCK=$bz -DNUMTRIALS=$t montecarlo.cu -o montecarlo
        ./montecarlo
    done
    printf "\n" >> montecarlo.csv
done

#clear files
rm -f montecarlo