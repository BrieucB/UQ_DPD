#!/usr/bin/env python3
import sys
sys.path.append('./model')

import korali
from mpi4py import MPI
from model.posteriorModel import *

# s={"Parameters":[1,2,3]}
# F(s, getReferencePoints())

# print(s["Reference Evaluations"])



e = korali.Experiment()

# Setting up the reference likelihood for the Bayesian Problem
e["Problem"]["Computational Model"] = lambda sampleData: F(sampleData, getReferencePoints())
e["Problem"]["Type"] = "Bayesian/Reference"
e["Problem"]["Likelihood Model"] = "Normal"
e["Problem"]["Reference Data"] = getReferenceData()

# Configuring CMA-ES parameters
e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = 12
e["Solver"]["Termination Criteria"]["Max Generations"] = 100

# Configuring the problem's random distributions
e["Distributions"][0]["Name"] = "Uniform 0"
e["Distributions"][0]["Type"] = "Univariate/Uniform"
e["Distributions"][0]["Minimum"] = 0.0
e["Distributions"][0]["Maximum"] = +100.0

# Configuring the problem's variables
e["Variables"][0]["Name"] = "a"
e["Variables"][0]["Prior Distribution"] = "Uniform 0"
e["Variables"][0]["Initial Value"] = +2.5
e["Variables"][0]["Initial Standard Deviation"] = +0.5

e["Variables"][1]["Name"] = "gamma"
e["Variables"][1]["Prior Distribution"] = "Uniform 0"
e["Variables"][1]["Initial Value"] = +10.0
e["Variables"][1]["Initial Standard Deviation"] = +0.5

e["Variables"][2]["Name"] = "[Sigma]"
e["Variables"][2]["Prior Distribution"] = "Uniform 0"
e["Variables"][2]["Initial Value"] = +2.5
e["Variables"][2]["Initial Standard Deviation"] = +0.5

# Configuring output settings
e["File Output"]["Path"] = '_korali_result_cmaes'
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