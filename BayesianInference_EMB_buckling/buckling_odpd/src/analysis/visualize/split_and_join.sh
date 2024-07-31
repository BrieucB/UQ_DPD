#!/bin/bash

#bash copy.sh

python3 SplitMBs.py -D output_emb -O output_equil

bash create_xyz_trj.sh

python3 all_analysis.py
