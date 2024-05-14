#!/usr/bin/env python3

# Importing the computational model
import json
import sys
sys.path.append('./model')

from model.plots import plot_credible_intervals
import korali
from model.posteriorModel import *
from mpi4py import MPI
import numpy as np

data = {}
data['X'] = getReferencePointsVisco()
data['Y'] = getReferenceDataVisco() 

# Evaluate the model for all the parameters from the previous step
e = korali.Experiment()

# Propagation to other densities
x = np.linspace(2.995, 3.004, 100)

e['Problem']['Type'] = 'Propagation'
e["Problem"]['Execution Model'] = lambda sampleData: viscosity_analytic_prop(sampleData, x)

# load the data from the sampling
with open('_setup/_korali_result_samples/latest') as f:
    d = json.load(f)

e['Variables'][0]['Name'] = 'rc'
v = [p[0] for p in d['Results']['Posterior Sample Database']]
e['Variables'][0]['Precomputed Values'] = v

e['Variables'][1]['Name'] = 'gamma'
v = [p[1] for p in d['Results']['Posterior Sample Database']]
e['Variables'][1]['Precomputed Values'] = v

# e['Variables'][2]['Name'] = 'power'
# v = [p[2] for p in d['Results']['Posterior Sample Database']]
# e['Variables'][2]['Precomputed Values'] = v

e['Variables'][2]['Name'] = '[Sigma]'
v = [p[2] for p in d['Results']['Posterior Sample Database']]
e['Variables'][2]['Precomputed Values'] = v

e['Solver']['Type'] = 'Executor'
e['Solver']['Executions Per Generation'] = 100

e['Console Output']['Verbosity'] = 'Minimal'
e['File Output']['Path'] = '_setup/_korali_result_propagation'
e['Store Sample Information'] = True

k = korali.Engine()

# k.setMPIComm(MPI.COMM_WORLD)
# k["Conduit"]["Type"] = "Distributed"
# k["Conduit"]["Ranks Per Worker"] = 2
k.run(e)

# Uncomment the next two lines to plot the credible intervals
#from plots import *
plot_credible_intervals('_setup/_korali_result_propagation/latest', data)
print(data)