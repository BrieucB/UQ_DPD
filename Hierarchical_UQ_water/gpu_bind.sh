#!/bin/bash                                                                                                                                                                                                        

RED='\033[0;31m'
GREEN='\033[0;32m'
PURPLE='\033[0;35m'
NC='\033[0m'


# Ranks[0,1] on GPU 0, ... Ranks[2k, 2k+1] on GPU k                                                                                                                                                                
# Run rank NPROC-1 without gpu                                                                                                                                                                                     

NODE=$(hostname)
NGPUS_NODE=$(nvidia-smi -L | wc -l)

#echo -e "Node $NODE contains $NGPUS_NODE GPUs."                                                                                                                                                                   

#export CUDA_VISIBLE_DEVICES=$(( ((OMPI_COMM_WORLD_RANK + 1 )/ 2) % $NGPUS_NODE))                                                                                                                                  
export CUDA_VISIBLE_DEVICES=$(( ((OMPI_COMM_WORLD_RANK)/ 2) % $NGPUS_NODE))

if [ $OMPI_COMM_WORLD_RANK ==  $(( OMPI_COMM_WORLD_SIZE - 1 )) ] # Last rank is the Korali engine                                                                                                                  
then
    export CUDA_VISIBLE_DEVICES=""
fi


echo -e "Global MPI rank ${RED} $OMPI_COMM_WORLD_RANK ${NC} running on node ${PURPLE} $NODE ${NC} under local rank $OMPI_COMM_WORLD_LOCAL_RANK runs on GPU $CUDA_VISIBLE_DEVICES" | sort

# Let numa find the best cpu (-b = numa balancing)                                                                                                                                                                 
numactl --all $@



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
