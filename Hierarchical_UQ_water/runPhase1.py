#!/usr/bin/env python3
import sys
sys.path.append('./model')
from posteriorModel import *
import korali
from mpi4py import MPI # Needed to assign a MPI comm to each simulation

eList = []
############### Experiment 1: viscosity ###############

e1 = korali.Experiment()

e1["Problem"]["Computational Model"] = lambda sampleData: measure_viscosity(sampleData, getReferencePointsVisco())
e1["Problem"]["Type"] = "Bayesian/Reference"
e1["Problem"]["Likelihood Model"] = "Normal"
e1["Problem"]["Reference Data"] = getReferenceDataVisco()

# Configuring the problem's random distributions
e1["Distributions"][0]["Name"] = "Uniform 0"
e1["Distributions"][0]["Type"] = "Univariate/Uniform"
e1["Distributions"][0]["Minimum"] = 0.0
e1["Distributions"][0]["Maximum"] = 2.0

e1["Distributions"][1]["Name"] = "Uniform 1"
e1["Distributions"][1]["Type"] = "Univariate/Uniform"
e1["Distributions"][1]["Minimum"] = 0.0
e1["Distributions"][1]["Maximum"] = 80.0

e1["Variables"][0]["Name"] = "a"
e1["Variables"][0]["Prior Distribution"] = "Uniform 0"

e1["Variables"][1]["Name"] = "gamma"
e1["Variables"][1]["Prior Distribution"] = "Uniform 1"

e1["Variables"][2]["Name"] = "[Sigma]"
e1["Variables"][2]["Prior Distribution"] = "Uniform 1"

e1["Solver"]["Type"] = "Sampler/TMCMC"
e1["Solver"]["Population Size"] = 1000
e1["Solver"]["Target Coefficient Of Variation"] = 0.6
e1["Solver"]["Covariance Scaling"] = 0.02

e1["File Output"]["Path"] = "_setup/results_phase_1/" + "viscosity"
e1["Console Output"]["Verbosity"] = "Detailed"
eList.append(e1)

############### Experiment 2: isothermal compressibility ###############

e2 = korali.Experiment()

e2["Problem"]["Computational Model"] = lambda sampleData: measure_compressibility(sampleData, getReferencePointsComp())
e2["Problem"]["Type"] = "Bayesian/Reference"
e2["Problem"]["Likelihood Model"] = "Normal"
e2["Problem"]["Reference Data"] = getReferenceDataComp()

# Configuring the problem's random distributions
e2["Distributions"][0]["Name"] = "Uniform 0"
e2["Distributions"][0]["Type"] = "Univariate/Uniform"
e2["Distributions"][0]["Minimum"] = 0.0
e2["Distributions"][0]["Maximum"] = 2.0

e2["Distributions"][1]["Name"] = "Uniform 1"
e2["Distributions"][1]["Type"] = "Univariate/Uniform"
e2["Distributions"][1]["Minimum"] = 0.0
e2["Distributions"][1]["Maximum"] = 80.0

e2["Variables"][0]["Name"] = "a"
e2["Variables"][0]["Prior Distribution"] = "Uniform 0"

e2["Variables"][1]["Name"] = "gamma"
e2["Variables"][1]["Prior Distribution"] = "Uniform 1"

e2["Variables"][1]["Name"] = "[Sigma]"
e2["Variables"][1]["Prior Distribution"] = "Uniform 1"

e2["Solver"]["Type"] = "Sampler/TMCMC"
e2["Solver"]["Population Size"] = 1000
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
k["Profiling"]["Detail"] = "Full"
k["Profiling"]["Frequency"] = 0.5

k.run(eList)