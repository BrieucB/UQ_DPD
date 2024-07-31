#!/bin/bash

mode=$1

simnum0="00001"
simnum=${2:-${simnum0}}

nranks=${3:-2}

echo "Simulation number: $simnum"
 
python3 parameters.py --simnum ${simnum}
mpirun -np ${nranks} python3 equil.py $mode --simnum ${simnum}
