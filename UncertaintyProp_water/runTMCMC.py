#!/usr/bin/env python3
import sys
sys.path.append('./model')
from posteriorModel import *
from model.units import *
import korali
from mpi4py import MPI # Needed to assign a MPI comm to each simulation

# Optimization parameters
params = np.loadtxt('metaparam.dat', skiprows=1) # pop_size, 
pop_size= int(params[0])

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Reference data setup
if (rank == 0):
    convertToDPDUnitsDensityVisco('data/data_density_viscosity.dat', units)
    convertToDPDUnitsDensitySpeed('data/data_density_speed.dat', units)
comm.Barrier()

e1 = korali.Experiment()

e1["Problem"]["Computational Model"] = lambda sampleData: viscosity_analytic(sampleData, getReferencePointsVisco())
e1["Problem"]["Type"] = "Bayesian/Reference"
e1["Problem"]["Likelihood Model"] = "Normal"
e1["Problem"]["Reference Data"] = getReferenceDataVisco()

# Configuring the problem's random distributions
e1["Distributions"][0]["Name"] = "Uniform_rc"
e1["Distributions"][0]["Type"] = "Univariate/Uniform"
e1["Distributions"][0]["Minimum"] = 1.
e1["Distributions"][0]["Maximum"] = 2.

e1["Distributions"][1]["Name"] = "Uniform_gamma"
e1["Distributions"][1]["Type"] = "Univariate/Uniform"
e1["Distributions"][1]["Minimum"] = 70.0
e1["Distributions"][1]["Maximum"] = 200.0

e1["Distributions"][2]["Name"] = "Uniform_power"
e1["Distributions"][2]["Type"] = "Univariate/Uniform"
e1["Distributions"][2]["Minimum"] = 0.0
e1["Distributions"][2]["Maximum"] = 0.3

e1["Distributions"][3]["Name"] = "Uniform_sigma"
e1["Distributions"][3]["Type"] = "Univariate/Uniform"
e1["Distributions"][3]["Minimum"] = 0.
e1["Distributions"][3]["Maximum"] = 30.

e1["Variables"][0]["Name"] = "rc"
e1["Variables"][0]["Prior Distribution"] = "Uniform_rc"

e1["Variables"][1]["Name"] = "gamma"
e1["Variables"][1]["Prior Distribution"] = "Uniform_gamma"

# e1["Variables"][2]["Name"] = "power"
# e1["Variables"][2]["Prior Distribution"] = "Uniform_power"

e1["Variables"][2]["Name"] = "sigma_c"
e1["Variables"][2]["Prior Distribution"] = "Uniform_sigma"

e1["Solver"]["Type"] = "Sampler/TMCMC"
e1["Solver"]["Population Size"] = 5000
e1["Solver"]["Target Coefficient Of Variation"] = 0.1
e1["Solver"]["Covariance Scaling"] = 0.02

e1["Solver"]["Termination Criteria"]["Target Annealing Exponent"] = 1.0

e1['Store Sample Information'] = True

e1["File Output"]["Path"] = "_setup/_korali_result_samples/"
e1["Console Output"]["Verbosity"] = "Detailed"

k = korali.Engine()

k.setMPIComm(MPI.COMM_WORLD)
k["Conduit"]["Type"] = "Distributed"
k["Conduit"]["Ranks Per Worker"] = 1

k.run(e1)
