#!/usr/bin/env python

import korali
import sys
sys.path.append('./model')
from units import *

############################################################
####################### USING MIRHEO #######################
############################################################

from mirheoViscosity import *
from mirheoCompressibility import *

def measure_compressibility(s,X):
  # read parameters from Korali
  a = s["Parameters"][0]
  gamma = s["Parameters"][1]
  sig = s["Parameters"][2]
  power = 0.25

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

  # read parameters from Korali
  a = s["Parameters"][0]
  gamma = s["Parameters"][1]
  sig = s["Parameters"][2]

  power = 0.25

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
  n_ref=0
  for Xi in X: # Loop over the reference points: here on density
    
    def quadratic_func(y, eta):
      return ((Xi*Fx*h)/(2.*eta))*y*(1.-y/h)
  
    # Export the simulation parameters
    simu_param={'m':1.0, 'nd':Xi, 'rc':1.0, 'L':L, 'Fx':Fx}
    dpd_param={'a':a, 'gamma':gamma, 'kBT':kBT_s, 'power':power}
    p={'simu':simu_param, 'dpd':dpd_param}

    # Set output file. Mirheo seems to attribute a random number to the output name, making it
    # difficult to find the output file. Here we specify the output folder to retrieve the file.
    folder = "velocities/"
    name = 'a%.2f_gamma%.2f_power%.2f_n%d/'%(a,gamma,power,n_ref)
    n_ref+=1

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

############################################################
############## USING ANALYTIC DPD EQUATIONS ################
############################################################

def kineticVisco(kBT, s, rho_s, rc, gamma, m):
    """
    s = power*2
    """
    A=(3*kBT*(s+1)*(s+2)*(s+3))/(16*np.pi*rho_s*(rc**3)*gamma)
    B=(16*np.pi*rho_s*(rc**3)*gamma)/(5*m*(s+1)*(s+2)*(s+3)*(s+4)*(s+5))
    return((A+B)/rho_s)

def soundCelerity(kBT, s, rho_s, rc, gamma, m, a):
    alpha=0.103
    return(np.sqrt(kBT/m + 2*alpha*rc**4*a*rho_s/m**2))

def speed_analytic(s,X):
  # read parameters from Korali
  a = s["Parameters"][0]
  gamma = s["Parameters"][1]
  #power = s["Parameters"][2]
  sig = s["Parameters"][2]
  
  power = 0.38
  rc = constants['rc'] # Cutoff radius of the DPD particles
  kBT_s = constants['kBT_s'] # Energy scale of the DPD particles
  m = constants['m'] # Mass of a DPD bead 

  s["Reference Evaluations"] = []
  s["Standard Deviation"] = []

  for rho_i in X:
    c = soundCelerity(kBT_s, 2.*power, rho_i, rc, gamma, m, a)
    s["Reference Evaluations"] += [c] 
    s["Standard Deviation"] += [sig]

def viscosity_analytic(s,X):
  
  # read parameters from Korali
  a = s["Parameters"][0]
  gamma = s["Parameters"][1]
  #power = s["Parameters"][2]
  sig = s["Parameters"][2]
  
  power = 0.38
  rc = constants['rc'] # Cutoff radius of the DPD particles
  kBT_s = constants['kBT_s'] # Energy scale of the DPD particles
  m = constants['m'] # Mass of a DPD bead 


  s["Reference Evaluations"] = []
  s["Standard Deviation"] = []

  for rho_i in X:
    visco = kineticVisco(kBT_s, 2.*power, rho_i, rc, gamma, m)
    s["Reference Evaluations"] += [visco] 
    s["Standard Deviation"] += [sig]

############################################################
##################### REFERENCE DATA #######################
############################################################

def getReferencePointsVisco():
  """
  Returns the density in DPD units
  """
  list_rho_s=np.loadtxt('data/data_density_viscosity_DPD.dat', skiprows=1)[:,0]
  return list(list_rho_s[-1:])

def getReferenceDataVisco():
  """
  Returns the viscosity in DPD units
  """
  list_eta_s=np.loadtxt('data/data_density_viscosity_DPD.dat', skiprows=1)[:,1]
  return list(list_eta_s[-1:])

def getReferencePointsSpeed():
  """
  Returns the density in DPD units
  """
  list_rho_s=np.loadtxt('data/data_density_speed_DPD.dat', skiprows=1)[:,0]
  return list(list_rho_s[2:3])

def getReferenceDataSpeed():
  """
  Returns the speed in DPD units
  """
  list_c_s=np.loadtxt('data/data_density_speed_DPD.dat', skiprows=1)[:,1]
  return list(list_c_s[2:3])

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
    X = getReferencePointsVisco()
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
      print('Reference compressibility:', getReferenceDataComp())
      print("rho:", X, "s[\"Reference Evaluations\"]:", s["Reference Evaluations"])
      print("s[\"Standard Deviation\"]:", s["Standard Deviation"][0])

if __name__ == '__main__':
    main(sys.argv[1:]) 