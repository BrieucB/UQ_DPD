#!/usr/bin/env python3
import sys
import os 
import korali
from mpi4py import MPI
import numpy as np

from posteriorBuckling import compute_pbuckling, convertToDPDUnits, getReferencePoints, getReferenceData
from buckling_odpd_korali.src.generate import generate_sim

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Optimization parameters
params   = np.loadtxt('metaparam.dat', skiprows=1) # pop_size, 
pop_size = int(params[0])

# Generate parameter folder for buckling simulations
source_buckling_path = 'buckling_odpd_korali/src/'
init_buckling_path = 'init_buckling/'

if rank == 0:
  # Create the folder for the buckling simulations
  os.system(f'mkdir -p {init_buckling_path}')

  # Copy the mesh file to the buckling simulation folder
  os.system(f'cp {source_buckling_path}emb.off {init_buckling_path}')

  # Generate the initial buckling simulation
  generate_sim(source_path  = source_buckling_path, 
              simu_path   = init_buckling_path,
              par         = [['buck', '25.0', '0.0', '1']],
              obj         = 'emb', 
              forward     = None, 
              hysteresis  = None, 
              parallel    = True, 
              g           = 1, 
              N           = 1, 
              first       = None, 
              numJobs     = 1)

  comm.Barrier()

  # Convert data into DPD units
  convertToDPDUnits('data/data_X_pbuckling.dat')
comm.Barrier()

if rank == 0:
      os.makedirs('logs', exist_ok=True)
      with open("logs/korali.log", "a") as f:
        f.write(f"[Korali setup] Number of ranks: {comm.Get_size()}\n")
        f.write(f"[Korali setup] Parameters: {params}\n")
        f.write(f"[Korali setup] Running the inference for:\n")
        f.write(f"[Korali setup] X-reference: {getReferencePoints()}\n")
        f.write(f"[Korali setup] Y-reference: {getReferenceData()}\n")

e = korali.Experiment()

# Setting up the reference likelihood for the Bayesian Problem
e["Problem"]["Computational Model"] = lambda sampleData: compute_pbuckling(sampleData, getReferencePoints())
e["Problem"]["Type"] = "Bayesian/Reference"
e["Problem"]["Likelihood Model"] = "Normal"
e["Problem"]["Reference Data"] = getReferenceData()

# Configuring TMCMC parameters
e["Solver"]["Type"] = "Sampler/TMCMC"
e["Solver"]["Population Size"] = pop_size
e["Solver"]["Target Coefficient Of Variation"] = 1.0
e["Solver"]["Covariance Scaling"] = 0.04

# Configuring the problem's random distributions
e["Distributions"][0]["Name"] = "Prior ka"
e["Distributions"][0]["Type"] = "Univariate/Uniform"
e["Distributions"][0]["Minimum"] = +420000.
e["Distributions"][0]["Maximum"] = +430000.

e["Distributions"][1]["Name"] = "Prior_sigma"
e["Distributions"][1]["Type"] = "Univariate/Uniform"
e["Distributions"][1]["Minimum"] = +0.0
e["Distributions"][1]["Maximum"] = +50.0

# Configuring the problem's variables
e["Variables"][0]["Name"] = "ka"
e["Variables"][0]["Prior Distribution"] = "Prior ka"
e["Variables"][0]["Initial Value"] = +426863.0

e["Variables"][1]["Name"] = "[Sigma]"
e["Variables"][1]["Prior Distribution"] = "Prior_sigma"
e["Variables"][1]["Initial Value"] = +10.0  

# Configuring output settings
e["File Output"]["Path"] = '_korali_result_tmcmc'
e["File Output"]["Frequency"] = 1
e["Console Output"]["Frequency"] = 1

# Storing sample information
e["Store Sample Information"] = True

k = korali.Engine()

k.setMPIComm(MPI.COMM_WORLD)
k["Conduit"]["Type"] = "Distributed"
k["Conduit"]["Ranks Per Worker"] = 2

# Profiling
# k["Profiling"]["Detail"] = "Full"
# k["Profiling"]["Frequency"] = 0.5

k.run(e)