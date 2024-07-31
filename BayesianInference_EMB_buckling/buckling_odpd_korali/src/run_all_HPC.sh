#!/bin/bash

main () {
  mkdir -p logs
  
  bash clean_all.sh
  
  python3 generate.py -p aii 100.0 100.0 1 --object "emb" --forward --first -g 4 -N 1
    
  sbatch run_HPC.sbatch
}

time main
