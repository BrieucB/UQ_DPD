#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np 
import trimesh
import yaml
import os, fnmatch
import argparse

######################################################
# set-up simulation type: equilibration or restart

parser = argparse.ArgumentParser()
parser.add_argument('--simnum', dest = 'simnum', default = '00001eq')
parser.add_argument('--par', dest = 'par', default = None)
args = parser.parse_args()

######################################################

filename_default = f'../../parameter/parameters-default{args.simnum}.yaml'
with open(filename_default, 'rb') as f:
    parameters_default = yaml.load(f, Loader = yaml.CLoader)

######################################################

par_list = []
par_path = '../../parameter/'
file_list = os.listdir(par_path)
file_list = np.sort(fnmatch.filter(file_list, f'parameters-default*'))
if(args.par):
    for file_par in file_list:
        filename_default = f'../../parameter/{file_par}'
        with open(filename_default, 'rb') as f:
            parameters_default = yaml.load(f, Loader = yaml.CLoader)
        par_list.append(parameters_default[args.par])
else:
    par_list = list(np.arange(len(file_list)))

sim_folder = np.sort(os.listdir('../../trj_eq/'))

print(sim_folder)

cnt = 0
vols = []
time = [] 
vol_time = []
cols = []
it = 0
for folder in sim_folder:
    new_color = 'blue'
    #if(folder.endswith('eq')):
    #    continue
    filepath = f'../../trj_eq/' + folder + "/"
    print(filepath)
    xyz_files = np.sort(os.listdir(filepath))
    xyz_files = fnmatch.filter(xyz_files, f'emb*.xyz')
    cnt = 0 
    simnum = folder[3:]
    filename_default = f'../../parameter/parameters-default{simnum}.yaml'
    with open(filename_default, 'rb') as f:
        parameters_default = yaml.load(f, Loader = yaml.CLoader)
    filename = f'../../parameter/parameters{simnum}.yaml'
    with open(filename, 'rb') as f:
        parameters = yaml.load(f, Loader = yaml.CLoader)
    objFile = '../../' + parameters_default["objFile"] 
    mesh = trimesh.load(objFile)
    #print(len(mesh.vertices))
    for xyz in xyz_files:
        r = np.loadtxt(filepath + xyz, skiprows = 2)
        #print(len(mesh.vertices), len(r))
        mesh.vertices[:,0] = r[:,1]
        mesh.vertices[:,1] = r[:,2]
        mesh.vertices[:,2] = r[:,3]
        vol_time.append(np.abs(mesh.volume))
        time.append(cnt)
        cnt += 1
    
    buck = parameters_default["buck"]
    aii = parameters_default["aii"]
    alfa = parameters_default["alpha"]
    rhow = parameters_default["rhow"]
    numsteps_eq = parameters_default["numsteps_eq"]
    dt_eq = parameters_default["dt_eq"]
    kbt = parameters["kbt"]
    
    nevery = parameters["nevery_eq"]
    dt = parameters_default["dt_eq"]
    
    rate = buck * aii * rhow**2 * alfa / numsteps_eq / dt_eq
    p0 = rhow * kbt + alfa * aii * rhow**2
    
    press = p0 + rate * np.array(time) * nevery * dt    
    
    plt.plot(press, vol_time, marker = 'o', linestyle = '-', label = '$Nv = $' + f'{len(mesh.vertices)}')
    plt.legend()
    time = [] 
    vol_time = [] 
    xyz = xyz_files[-1]
    r = np.loadtxt(filepath + xyz, skiprows = 2)
    mesh.vertices[:,0] = r[:,1]
    mesh.vertices[:,1] = r[:,2]
    mesh.vertices[:,2] = r[:,3]
    vols.append(np.abs(mesh.volume))
    if(it > 0 and par_list[it] - par_list[it-1] < 0):
        new_color = 'red'
    it += 1
    cols.append(new_color)

plt.savefig(f'volume_time.png')
plt.clf()
plt.scatter(par_list[0:len(vols)], vols, marker = 'o', color = cols, linestyle = '-')
plt.savefig(f'volume.png')

