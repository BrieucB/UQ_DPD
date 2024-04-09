#!/usr/bin/env python

#from DPD_water.model.mirheoModel_v1 import *
from mirheoModel import *
import korali
import sys

def F(s,T):
  import h5py
  from mpi4py import MPI
  from scipy.optimize import curve_fit
  import numpy as np
  import glob

  def quadratic_func(y, eta):
    return ((Fx*h)/(2.*eta))*y*(1.-y/h)

  # read parameters from Korali
  a = s["Parameters"][0]
  gamma = s["Parameters"][1]
  power = s["Parameters"][2]
  sig = s["Parameters"][3]

  # Read the MPI Comm assigned by Korali and feed it to the Mirheo simulation
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
     
  # Parameters of the simulation
  if standalone:
    params=np.loadtxt('../metaparam.dat', skiprows=1) # L, Fx, rho_s, kBT_s, tmax, pop_size
    L = int(params[0]) # Size of the simulation box in the x-direction
    h = L
    Fx = params[1] # Force applied in the x-direction to create the Poiseuille flow
    rho_s =  params[2] # Density of DPD particles
    kBT_s = params[3] # Energy scale of the DPD particles
    tmax = params[4] # Maximum simulation time (TODO: 
    # find a criteria to stop the simulation when stationary flow is reached)

  else:
    params=np.loadtxt('metaparam.dat', skiprows=1) # L, Fx, rho_s, kBT_s, tmax, pop_size
    L = int(params[0])
    h = L
    Fx = params[1]
    rho_s =  params[2]
    kBT_s = params[3]
    tmax = params[4]

  s["Reference Evaluations"] = []
  s["Standard Deviation"] = []
  s["error_fit"] = []

  for Ti in T: # Loop over the reference points: here only one point
    # Export the simulation parameters
    simu_param={'m':1.0, 'nd':rho_s, 'rc':1.0, 'L':L, 'Fx':Fx, 'tmax':tmax}
    dpd_param={'a':a, 'gamma':gamma, 'kBT':kBT_s, 'power':power}
    p={'simu':simu_param, 'dpd':dpd_param}

    # Set output file. Mirheo seems to attribute a random number to the output name, making it
    # difficult to find the output file. Here we specify the output folder to retrieve the file.
    folder = "velocities/"
    name = 'a%.2f_gamma%.2f_power%.2f/'%(a,gamma,power)

    # Run the simulation
    run_Poiseuille(p=p, ranks=(1,1,1), dump=False, comm=comm, out=(folder, name))
    
    # Collect the result of the simulation: the velocity profile averaged on
    # after the flow has reached a stationary state
    file = glob.glob(folder+name+'prof_*.h5')[0]
    f = h5py.File(file)

    # Compute the viscosity by fitting the velocity profile to a parabola
    M=np.mean(f['velocities'][:,:,:,0], axis=(0,2))
    data_neg=M[:L]
    data_pos=M[L:]
    data=0.5*(-data_neg+data_pos) # average the two half profiles 

    xmin=0.5
    xmax=L-0.5
    x=np.linspace(xmin, xmax, L)

    popt, pcov = curve_fit(quadratic_func, x, data)
    eta=popt[0]

    # Save the velocity profile if running standalone
    if standalone:
      out=np.concatenate([[a, gamma, Ti, eta], data])
      with open("velo_prof.csv", "w") as f:
        np.savetxt(f, out.reshape(1, out.shape[0]))

    # Output the result 
    s["Reference Evaluations"] += [eta] # Viscosity in simulation units
    s["Standard Deviation"] += [sig] # Viscosity in simulation units

    # Compute the error and store it: can be accessed after inference to check
    # the quality of the fit
    s["error_fit"] += [float(np.sqrt(np.diag(pcov))[0])]

def getReferenceData():
  return [25] # Reference data is the viscosity of water at 25°C

def getReferencePoints():
  import numpy as np
  params=np.loadtxt('metaparam.dat', skiprows=1) # L, Fx, rho_s, kBT_s, tmax, pop_size
  rho_s =  params[2]
  kBT_s = params[3]
  
  rho_water = 997 # kg/m^3 
  kb = 1.3805e-23 # S.I  
  T0 = 25 # °C

  ul = 35e-9/1.0 # real/simu : 35nm = standard length of a gas vesicle 
  um = rho_water*ul**3 / rho_s
  ue = kb*(T0+273.15) / kBT_s
  ut = np.sqrt(um*ul**2/ue)

  # viscosity is in kg . m^-1 . s^-1
  u_eta=(um/(ul*ut)) 
  u_real=0.001 # from mPa.s to Pa.s

  # Turn the real data into simulation units
  return [0.89*u_real/u_eta] # Reference data is the viscosity (0.89 mPa.s) at 25°C \approx 14.24 in simulation units

def main(argv):
  import argparse
  import numpy as np
  import shutil
  from mpi4py import MPI

  try:
    shutil.rmtree('./velocities')
    shutil.rmtree('./restart')
    shutil.rmtree('./h5')
  except:
     pass

  parser = argparse.ArgumentParser()
  parser.add_argument('--a', type=float, default=False)
  parser.add_argument('--gamma', type=float, default=False)
  parser.add_argument('--power', type=float, default=False)

  args = parser.parse_args(argv)

  s={"Parameters":[0,0,0,0.5]}
  s["Parameters"][0]=args.a
  s["Parameters"][1]=args.gamma
  s["Parameters"][2]=args.power

  T=[25]

  F(s,T)

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  if rank == 0:
    print("T:", T, "s[\"Reference Evaluations\"]:", s["Reference Evaluations"], "s[\"error_fit\"]:", 100*s["error_fit"][0]/s["Reference Evaluations"][0], "%")
    print("s[\"Standard Deviation\"]:", s["Standard Deviation"][0])

if __name__ == '__main__':
    main(sys.argv[1:]) 