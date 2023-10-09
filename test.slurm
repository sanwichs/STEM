#!/bin/bash
#SBATCH -p xahdnormal
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=dcu:2
#SBATCH -J stem
#SBATCH -o single-%j.out
#SBATCH -e single-%j.out


source /work/home/acrg91cvj4/miniconda3/etc/profile.d/conda.sh
conda activate stem
module switch compiler/dtk/22.10.1   
#export NCCL_DEBUG=info
#export NCCL_IB_HCA=mlx5
#export NCCL_SOCKET_IFNAME=ib0
#export HSA_FORCE_FINE_GRAIN_PCIE=1
#export OMP_NUM_THREADS=1

export HIP_VISIBLE_DEVICES=0

python test.py
