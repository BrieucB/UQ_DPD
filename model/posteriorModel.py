#!/usr/bin/env python

from mirheoModel import *
import korali

def F(s,T):
  import h5py
  from mpi4py import MPI
  from scipy.optimize import curve_fit

  L = 12
  h = L
  Fx = 0.5
  rho_s = 8.0
  kBT_s = 0.01

  def quadratic_func(y, eta):
    return ((Fx*h)/(2*eta))*y*(1-y/h)

  # read parameters from Korali
  a = s["Parameters"][0]
  gamma = s["Parameters"][1]
  sig = s["Parameters"][2]

  # Read the MPI Comm assigned by Korali to feed it to the Mirheo simulation
  # If running on stand alone, use standard MPI communicator
  try:
    comm = korali.getWorkerMPIComm()
    standalone = False
  except TypeError:
     comm = MPI.COMM_WORLD
     standalone = True
  # rank = comm.Get_rank()
  # size = comm.Get_size()
  # print(f"MPI Rank: {rank}/{size}")

  s["Reference Evaluations"] = []
  s["Standard Deviation"] = []
  s["error_fit"] = []

  for Ti in T:
    # Export the simulation parameters
    simu_param={'m':1.0, 'nd':rho_s, 'rc':1.0, 'L':L, 'Fx':Fx, 't_dump_every':0.01}
    #dpd_param={'a':a, 'gamma':gamma, 'kBT':kB*(Ti+273), 'power':0.5}
    dpd_param={'a':a, 'gamma':gamma, 'kBT':kBT_s, 'power':0.5}
    p={'simu':simu_param, 'dpd':dpd_param}

    # Set output file
    folder = "stress/"
    name = 'a%.2f_gamma%.2f_Ti%.2f'%(a,gamma,Ti)

    # Run the simulation
    run_Poiseuille(p=p, ranks=(1,1,1), dump=False, comm=comm, out=(folder, name))

    # Collect the result of the simulation
    f = h5py.File(folder+name+'00001.h5')

    # Compute the viscosity by fitting the velocity profile to a parabola
    M=np.mean(f['velocities'][:,:,:,0], axis=(0,2))
    data_neg=M[:L]
    data_pos=M[L:]
    data=0.5*(-data_neg+data_pos)

    xmin=0.5
    xmax=L-0.5

    x=np.linspace(xmin, xmax, L)

    popt, pcov = curve_fit(quadratic_func, x, data)
    eta=popt[0]

    if standalone:
      out=np.concatenate([[a, gamma, Ti, eta], data])
      with open("velo_prof.csv", "w") as f:
        np.savetxt(f, out.reshape(1, out.shape[0]))

    s["error_fit"] += [float(np.sqrt(np.diag(pcov))[0])]

    #UNITS
    rho_water = 997 # kg/m^3 
    kb = 1.3805e-23 # S.I  
    #T0 = 298.15 # K

    ul = 35e-9 # real/simu
    um = rho_water*ul**3 / rho_s
    ue = kb*(Ti+273.15) / kBT_s
    ut = np.sqrt(um*ul**2/ue)

    # viscosity is in kg . m^-1 . s^-1
    u_eta=(um/(ul*ut))

    #print("um:", um, "ue:", ue, "ut:", ut, "u_eta:", u_eta)

    # Output the result
    s["Reference Evaluations"] += [eta*u_eta]
    # s["Reference Evaluations"] += [eta]
    s["Standard Deviation"] += [sig]

def getReferenceData():
  import numpy as np
  data=np.loadtxt('data_T_µ.dat.csv')
  
  return [25] #list(data[::10,1])

def getReferencePoints():
  import numpy as np
  data=np.loadtxt('data_T_µ.dat.csv')
  return [0.001*0.89] #list(data[::10,0]) # change units to Pa.s by multiplying by 0.001


def main(argv):
  import argparse
  import numpy as np
  import shutil
  from mpi4py import MPI

  try:
    shutil.rmtree('./stress')
  except:
     pass

  parser = argparse.ArgumentParser()
  parser.add_argument('--a', type=float, default=False)
  parser.add_argument('--gamma', type=float, default=False)
  parser.add_argument('--sigma', type=float, default=False)

  args = parser.parse_args(argv)

  s={"Parameters":[0,0,0]}
  s["Parameters"][0]=args.a
  s["Parameters"][1]=args.gamma
  s["Parameters"][2]=args.sigma

  data=np.loadtxt('../data_T_µ.dat.csv')

  T=[25] #list(data[::10,0])

  F(s,T)

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  if rank == 0:
    print("T:", T, "s[\"Reference Evaluations\"]:", s["Reference Evaluations"], "s[\"error_fit\"]:", s["error_fit"])


if __name__ == '__main__':
    main(sys.argv[1:])