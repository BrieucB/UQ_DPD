#!/bin/bash                                                                                                                                                                                                        

#SBATCH --job-name=testUQ                                                                                                                                                                                          
#SBATCH --time=10-00:00:00                                                                                                                                                                                         
#SBATCH --nodes=2                                                                                                                                                                                                  
#SBATCH --ntasks=17                                                                                                                                                                                                
#SBATCH --nodelist=compute-3-18,compute-6-1 #,compute-6-2 #compute-3-18                                                                                                                                            
#SBATCH --partition=gpu                                                                                                                                                                                            
#SBATCH --output=logs/korali.log                                                                                                                                                                                   
#SBATCH --gres=gpu                                                                                                                                                                                                 

module load cuda
module load phdf5

host1=$(scontrol show hostname $SLURM_NODELIST | sed -n 1p)
host2=$(scontrol show hostname $SLURM_NODELIST | sed -n 2p)
# host3=$(scontrol show hostname $SLURM_NODELIST | sed -n 3p)                                                                                                                                                      

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/bbenvegnen/Programs/conda_env/korali/.local/lib64
export PYTHONPATH=${PYTHONPATH}:/home/bbenvegnen/Programs/conda_env/korali/.local/lib/python3.8/site-packages/

# This allocates a supplementary process for the Korali engine on the first requested node                                                                                                                         
cat <<EOF > hostfile                                                                                                                                                                                               
$host1 slots=8 max-slots=8                                                                                                                                                                                         
$host2 slots=7 max-slots=14                                                                                                                                                                                        
EOF                                                                                                                                                                                                                


mpirun -np 15 --map-by slot --hostfile hostfile ./gpu_bind.sh ./runTMCMC.py
#mpirun -np 17 ./gpu_bind.sh ./runTMCMC.py                                                                                                                                                                         

# Cancel just submitted job: scancel $( squeue -u $USER -h -o "%i" -t RUNNING) 