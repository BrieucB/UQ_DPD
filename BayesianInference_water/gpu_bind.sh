#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Run rank 0 on GPU 0
# Ranks[1,2] on GPU 1, ... Ranks[2k+1, 2k+2] on GPU k

export CUDA_VISIBLE_DEVICES=$(( (OMPI_COMM_WORLD_LOCAL_RANK + 1 )/ 2 ))
echo -e "${GREEN}[MPI]${NC} Rank $OMPI_COMM_WORLD_LOCAL_RANK runs on ${RED}GPU $CUDA_VISIBLE_DEVICES ${NC}"

# Let numa find the best cpu (-b = numa balancing)
numactl --all -b $@

################### DUMP #####################
#cpus=$OMPI_COMM_WORLD_LOCAL_RANK

#echo $OMPI_COMM_WORLD_LOCAL_RANK $CUDA_VISIBLE_DEVICES $cpus

# export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
# echo $OMPI_COMM_WORLD_LOCAL_RANK
# case $OMPI_COMM_WORLD_LOCAL_RANK in
# 0) cpus=0-2 ;;
# 1) cpus=3-5 ;;
# 2) cpus=6-8 ;;
# 3) cpus=9-11 ;;
# 4) cpus=12-14 ;;
# 5) cpus=15-17 ;;
# 6) cpus=18-20 ;;
# 7) cpus=21-23 ;;
# esac

#numactl --all --physcpubind=$cpus $@
#taskset -apc $cpus $@    
