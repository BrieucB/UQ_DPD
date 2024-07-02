#!/bin/bash
#SBATCH -w gpu10
#SBATCH -p GPU_queue
#SBATCH --mem=20G
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gres=gpu:3
#SBATCH --ntasks=5
#SBATCH --job-name=UQDPD
#SBATCH -o %x.out
#SBATCH -e %x-error.log

# Create remote folder on the target machine to copy the program
RESULTS_DIR="$SLURM_SUBMIT_DIR/output"
mkdir -p $RESULTS_DIR

echo "JOB ID: $SLURM_JOB_ID"
echo "Project dir: $SLURM_SUBMIT_DIR"
echo "Copying project files to /scratch partition in node $SLURMD_NODENAME."
cp -r $SLURM_SUBMIT_DIR/* /tmp/
echo "Copy Completed!"
cd /tmp

# Load modules for Korali + Mirheo
module load GSL/2.7-GCC-12.3.0
module load Python/3.11.3-GCCcore-12.3.0
module load Eigen/3.4.0-GCCcore-12.3.0
module load pybind11/2.11.1-GCCcore-12.3.0
module load HDF5/1.14.0-gompi-2023a
module load CUDA/12.3.0

cd UQ_DPD/BayesianInference_water

# Run the job
#mpirun -np 9 python3.11 runTMCMC.py
#mpirun -mca rmaps seq --rankfile rankfile --display-map python runTMCMC.py

#mpirun -np 9 -mca rmaps seq --display-map python runTMCMC.py
#srun -n 5 python runTMCMC.py
#mpirun -np 11 python runTMCMC.py
mpirun -np 5 ./gpu_bind.sh ./runTMCMC.py

# Copy back the results to local folder on Superfe
cp *.log *.txt *.dat *.csv *.out *.json rankfile logs/korali.log $RESULTS_DIR
cp -r _korali_result_TMCMC/ $RESULTS_DIR

module purge