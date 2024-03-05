#!/usr/bin/env python

import sys
# prefix='/usr/local/lib/python3/dist-packages'
# sys.path.append(prefix)

from shear_water import *
import korali

def model_shear(s,T):
  import h5py
  from mpi4py import MPI

  kB=1.380649e-23

  # read parameters from Korali
  a = s["Parameters"][0]
  gamma = s["Parameters"][1]
  sig = s["Parameters"][2]
  shear_rate=1
  result=[]

  # Read the MPI Comm assigned by Korali to feed it to the Mirheo simulation
  comm = korali.getWorkerMPIComm()
  rank = comm.Get_rank()
  size = comm.Get_size()
  # print(f"MPI Rank: {rank}/{size}")

  s["Reference Evaluations"] = []
  s["Standard Deviation"] = []
  for Ti in T:
    # Export the simulation parameters
    simu_param={'m':1.0, 'nd':3, 'rc':1, 'L':8, 'shear_rate':shear_rate, 't_dump_every':0.01}
    dpd_param={'a':a, 'gamma':gamma, 'kBT':kB*(Ti+273), 'power':0.5}
    p={'simu':simu_param, 'dpd':dpd_param}

    # Set output file
    folder = "/home/rio/Workspace/uq_force_field/shear_RBC_v2-20240221T122910Z-001/shear_RBC_v2/shear_RBC/stress/"
    name = 'a%.2f_gamma%.2f_Ti%.2f'%(a,gamma,Ti)
    run_shear_flow(p=p, ranks=(1,1,1), dump=False, comm=comm, out=(folder, name))

    # Collect the result of the simulation
    f = h5py.File(folder+name+'00000.h5')

    #tau=(np.mean(f['stresses'][:,:,:,3], axis=(0,2))[1]+np.mean(f['stresses'][:,:,:,3], axis=(0,2))[-2])/2
    tau=np.mean(np.mean(f['stresses'][:,:,:,3], axis=(0,2))[2:-2])
    µ=tau/shear_rate
    #print(µ)

    # Output the result
    s["Reference Evaluations"] += [µ]
    s["Standard Deviation"] += [sig]

def getReferenceData():
  import numpy as np
  data=np.loadtxt('/home/rio/Workspace/uq_force_field/shear_RBC_v2-20240221T122910Z-001/shear_RBC_v2/shear_RBC/data_T_µ.dat.csv')
  return list(data[::10,1])

def getReferencePoints():
  import numpy as np
  data=np.loadtxt('/home/rio/Workspace/uq_force_field/shear_RBC_v2-20240221T122910Z-001/shear_RBC_v2/shear_RBC/data_T_µ.dat.csv')
  return list(data[::10,0])
