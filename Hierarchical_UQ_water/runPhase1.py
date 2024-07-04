#!/usr/bin/env python3
import sys
import os
sys.path.append('./model')
from posteriorModel import *
from model.units import *

import korali
from mpi4py import MPI # Needed to assign a MPI comm to each simulation

eList = []
params = np.loadtxt('metaparam.dat', skiprows=1) # max_a max_gamma pop_size
max_a = params[0]
max_gamma = params[1]
pop_size = int(params[2])

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Reference data setup
if (rank == 0):
    convertToDPDUnitsDensityVisco('data/data_density_viscosity.dat', units)
    convertToDPDUnitsDensitySpeed('data/data_density_speed.dat', units)
    print('Target speed:', getReferencePointsSpeed(), getReferenceDataSpeed())
    print('Target viscosity:', getReferencePointsVisco(), getReferenceDataVisco())

    os.makedirs('logs', exist_ok=True)
    with open("logs/korali.log", "a") as f:
        f.write(f"[Korali setup] Number of ranks: {comm.Get_size()}\n")
        f.write(f"[Korali setup] Parameters: {params}\n")

comm.Barrier()

############### Experiment 1: viscosity ###############

e1 = korali.Experiment()

e1["Problem"]["Computational Model"] = lambda sampleData: computeViscosity(sampleData, getReferencePointsVisco())
e1["Problem"]["Type"] = "Bayesian/Reference"
e1["Problem"]["Likelihood Model"] = "Normal"
e1["Problem"]["Reference Data"] = getReferenceDataVisco()

# Configuring the problem's random distributions
e1["Distributions"][0]["Name"] = "Uniform a"
e1["Distributions"][0]["Type"] = "Univariate/Uniform"
e1["Distributions"][0]["Minimum"] = 150.0
e1["Distributions"][0]["Maximum"] = max_a

e1["Distributions"][1]["Name"] = "Uniform gamma"
e1["Distributions"][1]["Type"] = "Univariate/Uniform"
e1["Distributions"][1]["Minimum"] = 1
e1["Distributions"][1]["Maximum"] = max_gamma

e1["Distributions"][2]["Name"] = "Uniform_sig"
e1["Distributions"][2]["Type"] = "Univariate/Uniform"
e1["Distributions"][2]["Minimum"] = 0.0
e1["Distributions"][2]["Maximum"] = 50.0

e1["Variables"][0]["Name"] = "a"
e1["Variables"][0]["Prior Distribution"] = "Uniform a"

e1["Variables"][1]["Name"] = "gamma"
e1["Variables"][1]["Prior Distribution"] = "Uniform gamma"

# e1["Variables"][2]["Name"] = "power"
# e1["Variables"][2]["Prior Distribution"] = "Uniform_0_2"

e1["Variables"][2]["Name"] = "sigma_eta"
e1["Variables"][2]["Prior Distribution"] = "Uniform_sig"

e1["Solver"]["Type"] = "Sampler/TMCMC"
e1["Solver"]["Population Size"] = pop_size
e1["Solver"]["Target Coefficient Of Variation"] = 0.6
e1["Solver"]["Covariance Scaling"] = 0.02

e1["File Output"]["Path"] = "_setup/results_phase_1/" + "viscosity"
e1["Console Output"]["Verbosity"] = "Detailed"
eList.append(e1)

############### Experiment 2: speed of sound ###############

e2 = korali.Experiment()

e2["Problem"]["Computational Model"] = lambda sampleData: computeSpeed(sampleData, getReferencePointsSpeed())
e2["Problem"]["Type"] = "Bayesian/Reference"
e2["Problem"]["Likelihood Model"] = "Normal"
#e2["Problem"]["Reference Data"] = getReferenceDataComp()
e2["Problem"]["Reference Data"] = getReferenceDataSpeed()

# Configuring the problem's random distributions
e2["Distributions"][0]["Name"] = "Uniform a"
e2["Distributions"][0]["Type"] = "Univariate/Uniform"
e2["Distributions"][0]["Minimum"] = 150.0
e2["Distributions"][0]["Maximum"] = max_a

e2["Distributions"][1]["Name"] = "Uniform gamma"
e2["Distributions"][1]["Type"] = "Univariate/Uniform"
e2["Distributions"][1]["Minimum"] = 1
e2["Distributions"][1]["Maximum"] = max_gamma

e2["Distributions"][2]["Name"] = "Uniform_sig"
e2["Distributions"][2]["Type"] = "Univariate/Uniform"
e2["Distributions"][2]["Minimum"] = 0.1
e2["Distributions"][2]["Maximum"] = 50.0

e2["Variables"][0]["Name"] = "a"
e2["Variables"][0]["Prior Distribution"] = "Uniform a"

e2["Variables"][1]["Name"] = "gamma"
e2["Variables"][1]["Prior Distribution"] = "Uniform gamma"

# e2["Variables"][2]["Name"] = "power"
# e2["Variables"][2]["Prior Distribution"] = "Uniform_0_2"

e2["Variables"][2]["Name"] = "sigma_c"
e2["Variables"][2]["Prior Distribution"] = "Uniform_sig"

e2["Solver"]["Type"] = "Sampler/TMCMC"
e2["Solver"]["Population Size"] = pop_size
e2["Solver"]["Target Coefficient Of Variation"] = 0.6
e2["Solver"]["Covariance Scaling"] = 0.02

e2["File Output"]["Path"] = "_setup/results_phase_1/" + "compressibility"
e2["Console Output"]["Verbosity"] = "Detailed"
eList.append(e2)

# Starting Korali's Engine and running experiment
k = korali.Engine()

k.setMPIComm(MPI.COMM_WORLD)
k["Conduit"]["Type"] = "Distributed"
k["Conduit"]["Ranks Per Worker"] = 2

#Profiling
# k["Profiling"]["Detail"] = "Full"
# k["Profiling"]["Frequency"] = 0.5

k.run(eList)