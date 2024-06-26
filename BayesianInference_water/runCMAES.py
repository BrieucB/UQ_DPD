#!/usr/bin/env python3
import sys
sys.path.append('./model')

import korali
from mpi4py import MPI
from model.posteriorModel import *
import numpy as np

# Optimization parameters
params = np.loadtxt('metaparam.dat', skiprows=1) # L, Fx, rho_s, kBT_s, pop_size
pop_size = int(params[4])

e = korali.Experiment()

# Setting up the reference likelihood for the Bayesian Problem
e["Problem"]["Computational Model"] = lambda sampleData: F(sampleData, getReferencePoints())
e["Problem"]["Type"] = "Bayesian/Reference"
e["Problem"]["Likelihood Model"] = "Normal"
e["Problem"]["Reference Data"] = getReferenceData()

# Configuring CMA-ES parameters
e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = pop_size
e["Solver"]["Termination Criteria"]["Max Generations"] = 1000

# Configuring the problem's random distributions
e["Distributions"][0]["Name"] = "Prior_a"
e["Distributions"][0]["Type"] = "Univariate/Uniform"
e["Distributions"][0]["Minimum"] = +20.0
e["Distributions"][0]["Maximum"] = +60.0

e["Distributions"][1]["Name"] = "Prior_gamma"
e["Distributions"][1]["Type"] = "Univariate/Uniform"
e["Distributions"][1]["Minimum"] = +100.0
e["Distributions"][1]["Maximum"] = +200.0

e["Distributions"][2]["Name"] = "Prior_power"
e["Distributions"][2]["Type"] = "Univariate/Uniform"
e["Distributions"][2]["Minimum"] = +0.0
e["Distributions"][2]["Maximum"] = +1.0

e["Distributions"][3]["Name"] = "Prior_sigma"
e["Distributions"][3]["Type"] = "Univariate/Uniform"
e["Distributions"][3]["Minimum"] = +0.0
e["Distributions"][3]["Maximum"] = +100.0

# Configuring the problem's variables
e["Variables"][0]["Name"] = "a"
e["Variables"][0]["Prior Distribution"] = "Prior_a"
e["Variables"][0]["Initial Value"] = +20.0
e["Variables"][0]["Initial Standard Deviation"] = +10.0

e["Variables"][1]["Name"] = "gamma"
e["Variables"][1]["Prior Distribution"] = "Prior_gamma"
e["Variables"][1]["Initial Value"] = +50.0
e["Variables"][1]["Initial Standard Deviation"] = +20.0

e["Variables"][2]["Name"] = "power"
e["Variables"][2]["Prior Distribution"] = "Prior_power"
e["Variables"][2]["Initial Value"] = +0.5
e["Variables"][2]["Initial Standard Deviation"] = +0.1

e["Variables"][3]["Name"] = "[Sigma]"
e["Variables"][3]["Prior Distribution"] = "Prior_sigma"
e["Variables"][3]["Initial Value"] = +10
e["Variables"][3]["Initial Standard Deviation"] = +1.0

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