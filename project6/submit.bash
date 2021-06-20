#!/bin/bash
#SBATCH -J proj06
#SBATCH -A CS475-575
#SBATCH -p class
#SBATCH --gres=gpu:1
#SBATCH -o proj06.out
#SBATCH -e proj06.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=linxinw@oregonstate.edu


#Multiply
rm -f Multiply.csv

printf ",8,16,32,64,128,256,512\n" >> Multiply.csv
for dataset in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192
do
    printf "$dataset," >> Multiply.csv
    for localsize in 8 16 32 64 128 256 512
    do
        g++ -o Multiply -DNMB=$dataset -DLOCAL_SIZE=$localsize Multiply.cpp /usr/local/apps/cuda/10.1/lib64/libOpenCL.so.1.1 -lm -fopenmp
        ./Multiply
    done
    printf "\n" >> Multiply.csv
done

#clear files
rm -f Multiply



#MultiplyAdd
rm -f MultiplyAdd.csv
printf ",8,16,32,64,128,256,512\n" >> MultiplyAdd.csv
for dataset in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192
do
    printf "$dataset," >> MultiplyAdd.csv
    for localsize in 8 16 32 64 128 256 512
    do
        g++ -o MultiplyAdd -DNMB=$dataset -DLOCAL_SIZE=$localsize MultiplyAdd.cpp /usr/local/apps/cuda/10.1/lib64/libOpenCL.so.1.1 -lm -fopenmp
        ./MultiplyAdd
    done
    printf "\n" >> MultiplyAdd.csv
done

#clear files
rm -f MultiplyAdd




#MultuplyReduction
rm -f MultiplyReduction.csv
printf ",32,64,128,256\n" >> MultiplyReduction.csv
for dataset in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192
do
    printf "$dataset," >> MultiplyReduction.csv
    for localsize in 32 64 128 256
    do
        g++ -std=c++11 -o MultiplyReduction -DNMB=$dataset -DLOCAL_SIZE=$localsize MultiplyReduction.cpp /usr/local/apps/cuda/10.1/lib64/libOpenCL.so.1.1 -lm -fopenmp
        ./MultiplyReduction
    done
    printf "\n" >> MultiplyReduction.csv
done

#clear files
rm -f MultiplyReduction