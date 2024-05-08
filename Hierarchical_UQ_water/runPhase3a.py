#!/usr/bin/env python3

# Importing computational model
import korali
from mpi4py import MPI

# Creating hierarchical Bayesian problem from previous two problems
e = korali.Experiment()

psi = korali.Experiment()
psi.loadState('_setup/results_phase_2/latest')

e["Problem"]["Type"] = "Hierarchical/ThetaNew"
e["Problem"]["Psi Experiment"] = psi

e["Distributions"][0]["Name"] = "Uniform 0"
e["Distributions"][0]["Type"] = "Univariate/Uniform"
e["Distributions"][0]["Minimum"] = 0.0
e["Distributions"][0]["Maximum"] = 6.0

e["Distributions"][1]["Name"] = "Uniform 1"
e["Distributions"][1]["Type"] = "Univariate/Uniform"
e["Distributions"][1]["Minimum"] = 50.0
e["Distributions"][1]["Maximum"] = 150.0

# e["Distributions"][2]["Name"] = "Uniform 2"
# e["Distributions"][2]["Type"] = "Univariate/Uniform"
# e["Distributions"][2]["Minimum"] = 0.0
# e["Distributions"][2]["Maximum"] = 20.0

e["Variables"][0]["Name"] = "a"
e["Variables"][1]["Name"] = "gamma"
# e["Variables"][2]["Name"] = "Sigma"

e["Variables"][0]["Prior Distribution"] = "Uniform 0"
e["Variables"][1]["Prior Distribution"] = "Uniform 1"
# e["Variables"][2]["Prior Distribution"] = "Uniform 2"

e["Solver"]["Type"] = "Sampler/TMCMC"
e["Solver"]["Population Size"] = 10000
e["Solver"]["Target Coefficient Of Variation"] = 0.6
e["Solver"]["Covariance Scaling"] = 0.02
e["Solver"]["Burn In"] = 1
e["Solver"]["Max Chain Length"] = 1

e["Random Seed"] = 0xC0FFEE
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Path"] = "_setup/results_phase_3a/"

# Starting Korali's Engine and running experiment
k = korali.Engine()

k.setMPIComm(MPI.COMM_WORLD)
k["Conduit"]["Type"] = "Distributed"
k["Conduit"]["Ranks Per Worker"] = 1

k.run(e)