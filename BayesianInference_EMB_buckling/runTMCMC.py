#!/usr/bin/env python3
import sys
sys.path.append('./model')
import os 

import korali
from mpi4py import MPI
from model.posteriorModel import *
import numpy as np

# Optimization parameters
params   = np.loadtxt('metaparam.dat', skiprows=1) # pop_size, 
pop_size = int(params[0])

comm = MPI.COMM_WORLD
comm.Barrier()

rank = comm.Get_rank()
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