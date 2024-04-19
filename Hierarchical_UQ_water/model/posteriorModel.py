#!/usr/bin/env python

import korali
import sys
sys.path.append('./model')

from mirheoViscosity import *
from mirheoCompressibility import *

def measure_compressibility(s,X):
  # read parameters from Korali
  a = s["Parameters"][0]
  gamma = s["Parameters"][1]
  sig = s["Parameters"][2]
  power = 0.5

  # Read the MPI Comm assigned by Korali and feed it to the Mirheo simulation
  # If running on stand alone, use standard MPI communicator
  try:
    comm = korali.getWorkerMPIComm()
    standalone = False
  except TypeError:
     comm = MPI.COMM_WORLD
     standalone = True
     
  # Parameters of the simulation
  if standalone:
    params=np.loadtxt('../metaparam.dat', skiprows=1) # L, Fx, rho_s, kBT_s, pop_size
    L = int(params[0]) # Size of the simulation box in the x-direction
    h = L
    rho_s =  params[2] # Density of DPD particles
    kBT_s = params[3] # Energy scale of the DPD particles
    #print(X)

  else:
    params=np.loadtxt('metaparam.dat', skiprows=1) # L, Fx, rho_s, kBT_s, pop_size
    L = int(params[0])
    h = L
    rho_s =  params[2]
    kBT_s = params[3]

  s["Reference Evaluations"] = []
  s["Standard Deviation"] = []

  # Export the simulation parameters
  simu_param={'m':1.0, 'nd':rho_s, 'rc':1.0, 'L':L}
  dpd_param={'a':a, 'gamma':gamma, 'kBT':kBT_s, 'power':power}
  p={'simu':simu_param, 'dpd':dpd_param}

  # Set output file. Mirheo seems to attribute a random number to the output name, making it
  # difficult to find the output file. Here we specify the output folder to retrieve the file.
  folder = "virialstress/"
  name = 'a%.2f_gamma%.2f/'%(a,gamma)
  compressibility=getCompressibility(p=p, 
                                    ranks=(1,1,1), 
                                    comm=comm, 
                                    out=(folder, name))
  # Output the result 
  s["Reference Evaluations"] += [compressibility] 
  s["Standard Deviation"] += [sig]

def measure_viscosity(s,X):
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
  sig = s["Parameters"][2]

  power = 0.5

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
    params=np.loadtxt('../metaparam.dat', skiprows=1) # L, Fx, rho_s, kBT_s, pop_size
    L = int(params[0]) # Size of the simulation box in the x-direction
    h = L
    Fx = params[1] # Force applied in the x-direction to create the Poiseuille flow
    rho_s =  params[2] # Density of DPD particles
    kBT_s = params[3] # Energy scale of the DPD particles
    #print(X)

  else:
    params=np.loadtxt('metaparam.dat', skiprows=1) # L, Fx, rho_s, kBT_s, pop_size
    L = int(params[0])
    h = L
    Fx = params[1]
    rho_s =  params[2]
    kBT_s = params[3]

  s["Reference Evaluations"] = []
  s["Standard Deviation"] = []
  s["error_fit"] = []

  for Xi in X: # Loop over the reference points
    # Export the simulation parameters
    simu_param={'m':1.0, 'nd':Xi, 'rc':1.0, 'L':L, 'Fx':Fx}
    dpd_param={'a':a, 'gamma':gamma, 'kBT':kBT_s, 'power':power}
    p={'simu':simu_param, 'dpd':dpd_param}

    # Set output file. Mirheo seems to attribute a random number to the output name, making it
    # difficult to find the output file. Here we specify the output folder to retrieve the file.
    folder = "velocities/"
    name = 'a%.2f_gamma%.2f_power%.2f/'%(a,gamma,power)

    # Run the simulation
    run_Poiseuille(p=p, ranks=(1,1,1), dump=False, comm=comm, out=(folder, name))
    
    # Collect the result of the simulation: the velocity profile averaged
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
      out=np.concatenate([[a, gamma, Xi, eta], data])
      with open("velo_prof.csv", "w") as f:
        np.savetxt(f, out.reshape(1, out.shape[0]))

    # Output the result 
    s["Reference Evaluations"] += [eta] # Viscosity in simulation units
    s["Standard Deviation"] += [sig] # Viscosity in simulation units

    # Compute the error and store it: can be accessed after inference to check
    # the quality of the fit
    s["error_fit"] += [float(np.sqrt(np.diag(pcov))[0])]

def getReferencePoints():
  """
  Returns the density in DPD units
  """

  import numpy as np
  params=np.loadtxt('metaparam.dat', skiprows=1) # L, Fx, rho_s, kBT_s, pop_size
  
  rho_s = params[2]
  rho_water = 997 # density of water in kg/m^3 at 25°C  

  ul = 35e-9/1.0 # real/simu : 35nm = standard length of a gas vesicle 

  # We choose the standard mass scale to be defined by density of water at 25°C
  # divided by the standard density for DPD simulation 
  um = rho_water*ul**3 / rho_s
  
  rho_w_ref = 1.0e3*np.array([0.9982, 0.998, 0.9978, 0.9975, 0.9975, 0.997, 0.9968, 0.9965, 0.9962, 0.9959, 0.9956])
  list_rho_s = rho_w_ref*ul**3 / um

  return list_rho_s[::2] #[25] # Reference data is the viscosity of water at 25°C

def getReferenceData():
  """
  Returns the viscosity in DPD units
  """

  import numpy as np
  params=np.loadtxt('metaparam.dat', skiprows=1) # L, Fx, rho_s, kBT_s, pop_size
  rho_s = params[2]
  kBT_s = params[3]
  
  #rho_w_ref = np.array([0.9982, 0.998, 0.9978, 0.9975, 0.9975, 0.997, 0.9968, 0.9965, 0.9962, 0.9959, 0.9956])
  visco_ref = np.array([1.0016, 0.9775, 0.9544, 0.9321, 0.9107, 0.89, 0.8701, 0.8509, 0.8324, 0.8145, 0.7972])

  rho_water = 997 # kg/m^3 
  kb = 1.3805e-23 # S.I  
  T0 = 25 # °C

  ul = 35e-9/1.0 # real/simu : 35nm = standard length of a gas vesicle 
  um = rho_water*ul**3 / rho_s
  ue = kb*(T0+273.15) / kBT_s
  ut = np.sqrt(um*ul**2/ue)
  u_eta=um/(ul*ut)

  # viscosity is in kg . m^-1 . s^-1
  u_eta=um/(ul*ut)

  # real data is in mPa.s
  u_real=0.001 # from mPa.s to Pa.s

  # Turn the real data into simulation units
  return list(visco_ref*u_real/u_eta)[::2] #[0.89*u_real/u_eta] # Reference data is the viscosity (0.89 mPa.s) at 25°C \approx 14.24 in simulation units

def main(argv):
  import argparse
  import numpy as np
  import shutil
  from mpi4py import MPI

  try:
    shutil.rmtree('./velocities')
    shutil.rmtree('./restart')
    shutil.rmtree('./h5')
    shutil.rmtree('./virialstress')
    
  except:
     pass

  parser = argparse.ArgumentParser()
  parser.add_argument('--a', type=float, default=False)
  parser.add_argument('--gamma', type=float, default=False)
  parser.add_argument('--experiment', type=str, default=False)


  args = parser.parse_args(argv)

  s={"Parameters":[0,0,0.5]}
  s["Parameters"][0]=args.a
  s["Parameters"][1]=args.gamma

  if(args.experiment == 'viscosity'):
    X = getReferencePoints()
    measure_viscosity(s,X)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
      print("rho:", X, "s[\"Reference Evaluations\"]:", s["Reference Evaluations"], "s[\"error_fit\"]:", 100*np.array(s["error_fit"])/np.array(s["Reference Evaluations"]), "%")
      print("s[\"Standard Deviation\"]:", s["Standard Deviation"][0])

  elif(args.experiment == 'compressibility'):
    X = np.loadtxt('data_compressibility.dat', skiprows=1)[0]
    measure_compressibility(s,X)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
      print("rho:", X, "s[\"Reference Evaluations\"]:", s["Reference Evaluations"])
      print("s[\"Standard Deviation\"]:", s["Standard Deviation"][0])

if __name__ == '__main__':
    main(sys.argv[1:]) 