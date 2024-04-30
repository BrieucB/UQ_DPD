#!/usr/bin/env python3

# Importing computational model
import sys
import os
import korali

# Creating hierarchical Bayesian problem from previous two problems
e = korali.Experiment()

e["Problem"]["Type"] = "Hierarchical/Psi"

subProblem1 = korali.Experiment()
subProblem1.loadState('_setup/results_phase_1/' + 'compressibility' + '/latest')
e["Problem"]["Sub Experiments"][0] = subProblem1

#print(e["Problem"]["Sub Experiments"][0]["Variables"][1]["Name"])

subProblem2 = korali.Experiment()
subProblem2.loadState('_setup/results_phase_1/' + 'viscosity' + '/latest')
e["Problem"]["Sub Experiments"][1] = subProblem2

# Add probability of theta given psi, one per subproblem variable.
# theta = [a, gamma, power]
# psi = [mu_a, sig_a, mu_gamma, sig_gamma, mu_power, sig_power]
e["Variables"][0]["Name"] = "mu_a"
e["Variables"][1]["Name"] = "sig_a"
e["Variables"][2]["Name"] = "mu_gamma"
e["Variables"][3]["Name"] = "sig_gamma"
e["Variables"][4]["Name"] = "mu_power"
e["Variables"][5]["Name"] = "sig_power"

# Configuring the conditionnal distributions
e["Distributions"][0]["Name"] = "Conditional 0"
e["Distributions"][0]["Type"] = "Univariate/Laplace"
e["Distributions"][0]["Mean"] = "mu_a"
e["Distributions"][0]["Width"] = "sig_a"
#e["Distributions"][0]["Standard Deviation"] = "sig_a"

# e["Distributions"][0]["Minimum"] = 0.1
# e["Distributions"][0]["Maximum"] = 300.0

e["Distributions"][1]["Name"] = "Conditional 1"
e["Distributions"][1]["Type"] = "Univariate/Normal"
e["Distributions"][1]["Mu"] = "mu_gamma"
e["Distributions"][1]["Sigma"] = "sig_gamma"
e["Distributions"][1]["Minimum"] = 1.0
e["Distributions"][1]["Maximum"] = 300.0

e["Distributions"][2]["Name"] = "Conditional 2"
e["Distributions"][2]["Type"] = "Univariate/LogNormal"
e["Distributions"][2]["Mu"] = "mu_power"
e["Distributions"][2]["Sigma"] = "sig_power"
e["Distributions"][2]["Minimum"] = 0.01
e["Distributions"][2]["Maximum"] = 5.0

# Configuring the priors distributions for the hyperparameters
# Prior of mu_a
e["Distributions"][3]["Name"] = "Uniform 0"
e["Distributions"][3]["Type"] = "Univariate/Uniform"
e["Distributions"][3]["Minimum"] = 1.0
e["Distributions"][3]["Maximum"] = 50.0

# Prior of sig_a
e["Distributions"][4]["Name"] = "Uniform 1"
e["Distributions"][4]["Type"] = "Univariate/Uniform"
e["Distributions"][4]["Minimum"] = 0.1
e["Distributions"][4]["Maximum"] = 20.0

# Prior of mu_gamma
e["Distributions"][5]["Name"] = "Uniform 2"
e["Distributions"][5]["Type"] = "Univariate/Uniform"
e["Distributions"][5]["Minimum"] = 1.0
e["Distributions"][5]["Maximum"] = 200.0

# Prior of sig_gamma
e["Distributions"][6]["Name"] = "Uniform 3"
e["Distributions"][6]["Type"] = "Univariate/Uniform"
e["Distributions"][6]["Minimum"] = 1
e["Distributions"][6]["Maximum"] = 50.0

# Prior of mu_power
e["Distributions"][7]["Name"] = "Uniform 4"
e["Distributions"][7]["Type"] = "Univariate/Uniform"
e["Distributions"][7]["Minimum"] = 0.1
e["Distributions"][7]["Maximum"] = 2.0

# Prior of sig_power
e["Distributions"][8]["Name"] = "Uniform 5"
e["Distributions"][8]["Type"] = "Univariate/Uniform"
e["Distributions"][8]["Minimum"] = 0.1
e["Distributions"][8]["Maximum"] = 0.5

e["Variables"][0]["Prior Distribution"] = "Uniform 0"
e["Variables"][1]["Prior Distribution"] = "Uniform 1"
e["Variables"][2]["Prior Distribution"] = "Uniform 2"
e["Variables"][3]["Prior Distribution"] = "Uniform 3"
e["Variables"][4]["Prior Distribution"] = "Uniform 4"
e["Variables"][5]["Prior Distribution"] = "Uniform 5"

e["Problem"]["Conditional Priors"] = ["Conditional 0", "Conditional 1", "Conditional 2"]


# Configuring solver
e["Solver"]["Type"] = "Sampler/TMCMC"
e["Solver"]["Population Size"] = 2000
e["Solver"]["Burn In"] = 1
e["Solver"]["Target Coefficient Of Variation"] = 0.6
e["Solver"]["Covariance Scaling"] = 0.1

e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Path"] = "_setup/results_phase_2/"

# Starting Korali's Engine and running experiment
k = korali.Engine()
k.run(e)