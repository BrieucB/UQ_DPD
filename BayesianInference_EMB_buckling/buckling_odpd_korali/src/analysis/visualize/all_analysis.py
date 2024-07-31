#! /usr/bin/env python3

import MDAnalysis as mda
import numpy as np
import os, fnmatch
import warnings
from MDAnalysis.analysis import pca
import yaml 
from timeit import default_timer as timer

start = timer()

warnings.warn('ignore')

filename_default = '../../parameter/parameters-default00001eq.yaml'
with open(filename_default, 'rb') as f:
    parameters_default = yaml.load(f, Loader = yaml.CLoader)

default_count = parameters_default["numObjects"]
objFile = '../../' + parameters_default["objFile"]

path = './output_equil/'

import trimesh 

mesh = trimesh.load('sphere_thomson.off')

vertices = mesh.vertices

file0 = open('xyz0.xyz', 'w')

file0.write(f'{len(vertices)}\n')
file0.write('# generated using trimesh\n')

nicle = np.zeros(len(vertices))

cols = np.column_stack((nicle, vertices[:,0], vertices[:,1], vertices[:,2]))

np.savetxt(file0, cols)

xyz_list = list(np.sort(fnmatch.filter(os.listdir(path), 'MB_0*.xyz')))

for i in range(len(xyz_list)):
    xyz_list[i] = path + xyz_list[i]
#xyz_list[1:-1]

trj = mda.Universe(xyz_list[0], 'positions.xyz', format="XYZ", dt=1)
#trj.select_atoms('all').masses = 1.0

# for large number of files increase the number of files that can be opened
# ulimit -S -n 4096  #4096 is the HARD limit
#print(xyz_list)

###########################################################
print('Step:::::Calculating average/reference structure')
trajs = len(trj.trajectory)
vertices = len(trj.trajectory[0].positions)
av = 0
for i in range(trajs):
    av += trj.trajectory[i].positions / trajs

f = open('ref.xyz', 'w')
f.write(f'{vertices}\n')
f.write(f'# generated in MDAnalysis\n')    
cols = np.column_stack((nicle, av))
np.savetxt(f, cols)
f.close()


###########################################################


print('Step:::::Performing trajectory alignment using RMSD')

from MDAnalysis.analysis import align
#from MDAnalysis.analysis.rms import rmsd

ref = mda.Universe('ref.xyz', format="XYZ")
#ref = mda.Universe(xyz_list[0], format="XYZ")

alignment = align.AlignTraj(trj, ref, filename='rmsfit.xyz')

alignment.run()


###########################################################
