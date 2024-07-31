#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np 
import yaml
import os
import argparse

######################################################
# set-up simulation type: equilibration or restart

parser = argparse.ArgumentParser()
parser.add_argument('--simnum', dest = 'simnum', default = '00001eq')
args = parser.parse_args()

######################################################

filename_default = f'../parameter/parameters-default{args.simnum}.yaml'
with open(filename_default, 'rb') as f:
    parameters_default = yaml.load(f, Loader = yaml.CLoader)

Lx = parameters_default["Lx"]
Ly = parameters_default["Ly"]
Lz = parameters_default["Lz"]

vol = Lx * Ly * Lz
######################################################

sim_folder = np.sort(os.listdir('../pressure/'))

for folder in sim_folder:
    #if(folder.endswith('eq')):
    #   continue
    filepath = f'../pressure/' + folder + "/water.csv"
    print(filepath)
    #time, pressure = 
    a = np.loadtxt(filepath, skiprows = 1, delimiter = ',')
    time = a[:,0]
    pressure = a[:,1] / vol
    plt.plot(time, pressure, marker = 'o', color = 'b', linestyle = '-')

plt.savefig(f'pressure_time.png')
