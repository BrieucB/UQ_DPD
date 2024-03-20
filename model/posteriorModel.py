#!/usr/bin/env python

from mirheoModel import *
import korali

def F(s,T):
  import h5py
  from mpi4py import MPI
  from scipy.optimize import curve_fit

  L=8
  h=L
  Fx=0.5

  def quadratic_func(y, eta):
      return ((Fx*h)/(2*eta))*y*(1-y/h)

  kB=1.380649e-23

  # read parameters from Korali
  a = s["Parameters"][0]
  gamma = s["Parameters"][1]
  sig = s["Parameters"][2]
  shear_rate=1

  # Read the MPI Comm assigned by Korali to feed it to the Mirheo simulation
  comm = korali.getWorkerMPIComm()
  rank = comm.Get_rank()
  size = comm.Get_size()
  # print(f"MPI Rank: {rank}/{size}")

  s["Reference Evaluations"] = []
  s["Standard Deviation"] = []

  for Ti in T:
    # Export the simulation parameters
    simu_param={'m':1.0, 'nd':8, 'rc':1, 'L':L, 'shear_rate':shear_rate, 't_dump_every':0.01}
    dpd_param={'a':a, 'gamma':gamma, 'kBT':kB*(Ti+273), 'power':0.5}
    p={'simu':simu_param, 'dpd':dpd_param}

    # Set output file
    folder = "/home/rio/Workspace/uq_force_field/DPD_water/stress/"
    name = 'a%.2f_gamma%.2f_Ti%.2f'%(a,gamma,Ti)
    run_Poiseuille(p=p, ranks=(1,1,1), dump=False, comm=comm, out=(folder, name))

    # Collect the result of the simulation
    f = h5py.File(folder+name+'00001.h5')

    #print(f)
    M=np.mean(f['velocities'][:,:,:,0], axis=(0,2))
    data_neg=M[:L]
    data_pos=M[L:]
    data=0.5*(-data_neg+data_pos)

    xmin=0.5
    xmax=L-0.5

    x=np.linspace(xmin, xmax, L)

    popt, _ = curve_fit(quadratic_func, x, data)

    eta=popt[0]

    # Output the result
    s["Reference Evaluations"] += [eta]
    s["Standard Deviation"] += [sig]

def getReferenceData():
  import numpy as np
  data=np.loadtxt('/home/rio/Workspace/uq_force_field/shear_RBC_v2-20240221T122910Z-001/shear_RBC_v2/shear_RBC/data_T_µ.dat.csv')
  return list(data[::10,1])

def getReferencePoints():
  import numpy as np
  data=np.loadtxt('/home/rio/Workspace/uq_force_field/shear_RBC_v2-20240221T122910Z-001/shear_RBC_v2/shear_RBC/data_T_µ.dat.csv')
  return list(data[::10,0])
