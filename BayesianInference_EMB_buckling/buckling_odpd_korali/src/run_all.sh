#!/bin/bash

main () {
  mkdir -p logs
  
  bash clean_all.sh
  
  python3 generate.py -p buck 25.0 0.0 4 --object "emb" --parallel
  
  bash commands.txt
}

time main
