#!/bin/bash

mkdir -p _setup/plots

mpirun -np 12 python runPhase1.py 

python3 -m korali.plot --dir _setup/results_phase_1/compressibility/ --output _setup/plots/phase1_speed.png
python3 -m korali.plot --dir _setup/results_phase_1/viscosity/ --output _setup/plots/phase1_visco.png

mpirun -np 12 python runPhase2.py 
python3 -m korali.plot --dir _setup/results_phase_2/ --output _setup/plots/phase2.png

mpirun -np 12 python runPhase3a.py 
python3 -m korali.plot --dir _setup/results_phase_3a/ --output _setup/plots/phase3a.png