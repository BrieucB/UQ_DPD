#!/usr/bin/bash

dir0=Fx_dep
mkdir $dir0
cd $dir0

cp ../runBatchJolly.sh .

for Fx in 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.010 0.020 0.030 0.040 0.050 0.060 0.070 0.080 0.090
#0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0
do
    dir=Fx$Fx
    mkdir $dir
    cd $dir

    cat <<EOF > ../../metaparam.dat
#L Fx rho_s kBT_s pop_size
19 $Fx 3.0 0.01 500
EOF

    mpirun -np 2 python3 ../../posteriorModel.py --power=0.125 --gamma=40 --a=0.1 > log.txt
    cd ..
done
