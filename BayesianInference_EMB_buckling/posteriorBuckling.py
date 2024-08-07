#!/usr/bin/env python

import fnmatch
import os

from matplotlib import pyplot as plt
import korali
import sys
import numpy as np

import trimesh
import yaml

from buckling_odpd_korali.src.generate import generate_sim
from buckling_odpd_korali.src.equil import run_equil
from buckling_odpd_korali.src.parameters import write_parameters

dump = False

def compute_pbuckling(sample,X):
  from mpi4py import MPI

  # Get source path
  source_path = 'buckling_odpd_korali/src/'
  
  # Read parameters to be optimized from Korali
  ka, sig = sample["Parameters"]
  #ka = 597609.5617529879

  # Read the MPI Comm assigned by Korali and feed it to the Mirheo simulation
  # If running on stand alone, use standard MPI communicator
  try:
    comm       = korali.getWorkerMPIComm()
    standalone = False
  except TypeError:
     comm       = MPI.COMM_WORLD
     standalone = True

  rank = comm.Get_rank()

  sample["Reference Evaluations"] = []
  sample["Standard Deviation"]    = []

  n_ref=0
  for Xi in X: # Loop over the reference points: here on density

    # Set simu path
    folder    = "out/"
    name      = 'ka%.2f_n%d/'%(ka,n_ref)
    simu_path = folder + name
    simnum    = '%05d'%1
    n_ref    += 1
    os.system(f'mkdir -p {simu_path}')

    # Prepare the simulation
    if rank == 0:
      generate_sim(source_path  = source_path, 
                    simu_path   = simu_path,
                    par         = [['buck', '25.0', '0.0', '1']], #None, 
                    obj         = 'emb', 
                    forward     = None, 
                    hysteresis  = None, 
                    parallel    = True, 
                    g           = 1, 
                    N           = 1, 
                    first       = None, 
                    numJobs     = 1)
      
      write_parameters(source_path  = source_path, 
                        simu_path   = simu_path,
                        simnum      = simnum)
      
    comm.Barrier()

    # Run the simulation
    run_equil(source_path = source_path, 
              simu_path   = simu_path,
              simnum      = simnum,
              equil       = True,
              restart     = False,
              comm        = comm)
    
    # Read the simulation parameters
    filename_default = simu_path + 'parameter/parameters-default' + simnum + '.yaml'
    with open(filename_default, 'rb') as f:
        parameters_default = yaml.load(f, Loader = yaml.CLoader)

    filename = simu_path + f'parameter/parameters{simnum}.yaml'
    with open(filename, 'rb') as f:
        parameters = yaml.load(f, Loader = yaml.CLoader)

    # Loop over the meshes and compute the volume
    xyzpath   = simu_path + f"/trj_eq/sim{simnum}/"
    xyz_files = np.sort(os.listdir(xyzpath))
    xyz_files = fnmatch.filter(xyz_files, f'emb*.xyz')

    time     = [] 
    vol_time = []
    cnt      = 0
    mesh     = trimesh.load_mesh(source_path + parameters_default["objFile"] )

    for xyz in xyz_files:
        r = np.loadtxt(xyzpath + xyz, skiprows = 2)
        mesh.vertices[:,0] = r[:,1]
        mesh.vertices[:,1] = r[:,2]
        mesh.vertices[:,2] = r[:,3]
        vol_time.append(np.abs(mesh.volume))
        time.append(cnt)
        cnt += 1
    
    # Grab the parameters to compute the buckling pressure 
    buck         = parameters_default["buck"]
    aii          = parameters_default["aii"]
    alfa         = parameters_default["alpha"]
    rhow         = parameters_default["rhow"]
    numsteps_eq  = parameters_default["numsteps_eq"]
    dt_eq        = parameters_default["dt_eq"]
    kbt          = parameters["kbt"]    
    nevery       = parameters["nevery_eq"]
    dt           = parameters_default["dt_eq"]
    
    rate  = buck * aii * rhow**2 * alfa / numsteps_eq / dt_eq

    p0 = rhow * kbt + alfa * aii * rhow**2
    press_0 = p0 + 2 * rate * np.array(time)
    press_tmp = (p0 + buck * aii * rhow**2 * alfa) * np.ones(len(press_0))
    press = np.minimum(press_0, press_tmp)
    
    # Plot the volume vs pressure
    if(dump == True):
      plt.plot(press, vol_time, marker = 'o', linestyle = '-', label = '$Nv = $' + f'{len(mesh.vertices)}')
      plt.legend()
      plt.savefig(simu_path + f'volume_time.png')
      plt.close()

    pbuckling = press[np.where(vol_time < 0.95*vol_time[0])[0][0]]
    print('pbuckling =', pbuckling)

    # Output the result 
    sample["Reference Evaluations"] += [pbuckling] 
    sample["Standard Deviation"]    += [sig]

    # Clean up simulation files
    os.system(f'rm -rf {simu_path}restart/')    
    os.system(f'rm -f {simu_path}commands.txt')
    os.system(f'rm -f {simu_path}posq.txt')
    os.system(f'rm -f {simu_path}run_HPC.sbatch')

    if dump == False:
      os.system(f'rm -rf {simu_path}trj_eq/')
      os.system(f'rm -f {simu_path}volume_time.png')