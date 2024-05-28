#!/usr/bin/env python

import korali
import sys
sys.path.append('./model')
import numpy as np

from units import *
from mirheoModel import *

def kineticVisco(kBT, s, rho_s, rc, gamma, m):
    """
    s = power*2
    """
    A=(3*kBT*(s+1)*(s+2)*(s+3))/(16*np.pi*(rho_s**2)*(rc**3)*gamma)
    B=(16*np.pi*(rc**3)*gamma)/(5*m*(s+1)*(s+2)*(s+3)*(s+4)*(s+5))
    return(A+B)

def soundCelerity(kBT, s, rho_s, rc, gamma, m, a):
    alpha=0.103
    return(np.sqrt(kBT/m + 2*alpha*(rc**4)*a*rho_s/(m**2)))

############################################################
##################### VELOCITY FCTS ########################
############################################################

def speed_analytic(s,X):
    # read parameters from Korali
    rc, gamma, sig = s["Parameters"]
    power = 0.12

    kBT_s = constants['kBT_s'] # Energy scale of the DPD particles
    m = constants['m'] # Mass of a DPD bead

    s["Reference Evaluations"] = []
    s["Standard Deviation"] = []

    for rho_i in X:
        speed = soundCelerity(kBT_s, 2.*power, rho_i, rc, gamma, m)
        s["Reference Evaluations"] += [speed] 
        s["Standard Deviation"] += [sig]

def speed_analytic_prop(s,X):
    rc, gamma, sig = s["Parameters"]
    power = 0.12
    
    kBT_s = constants['kBT_s'] # Energy scale of the DPD particles
    m = constants['m'] # Mass of a DPD bead

    s['X'] = X.tolist()
    s['sigma'] = sig
    s['Evaluations'] = []
    for rho_i in X:
        speed = soundCelerity(kBT_s, 2.*power, rho_i, rc, gamma, m)
        s['Evaluations'] += [speed]

############################################################
##################### VISCOSITY FCTS #######################
############################################################

def viscosity_simu_prop(s,X):
  import h5py
  from mpi4py import MPI
  from scipy.optimize import curve_fit
  import glob
  import shutil

  # read parameters from Korali
  print(s["Parameters"])
  a, gamma, power, sig = s["Parameters"]

  # Read the MPI Comm assigned by Korali and feed it to the Mirheo simulation
  # If running on stand alone, use standard MPI communicator
  try:
    comm = korali.getWorkerMPIComm()
    standalone = False
  except TypeError:
     comm = MPI.COMM_WORLD
     standalone = True

  rank = comm.Get_rank()

  L = constants['L'] # Size of the simulation box in the x-direction
  h = L
  Fx = constants['Fx'] # Force applied in the x-direction to create the Poiseuille flow
  kBT_s = constants['kBT_s'] # Energy scale of the DPD particles
  m = constants['m'] # Mass of a DPD bead
  rc = constants['rc'] # Cutoff radius of the DPD particles

  s['X'] = X.tolist()
  s['sigma'] = sig
  s['Evaluations'] = []

  n_ref=0
  for Xi in X: # Loop over the reference points: here on density
    
    def quadratic_func(y, eta):
      return ((Xi*Fx*h)/(2.*eta))*y*(1.-y/h)
  
    # Export the simulation parameters
    simu_param={'m':m, 'nd':Xi, 'rc':rc, 'L':L, 'Fx':Fx}
    dpd_param={'a':a, 'gamma':gamma, 'kBT':kBT_s, 'power':power}
    p={'simu':simu_param, 'dpd':dpd_param}

    # Set output file. Mirheo seems to attribute a random number to the output name, making it
    # difficult to find the output file. Here we specify the output folder to retrieve the file.
    folder = "velocities/"
    name = 'a%.2f_gamma%.2f_power%.2f_n%d/'%(a,gamma,power,n_ref)
    n_ref+=1

    # Log the run
    # if rank == 0:
    #   with open("logs/korali.log", "a") as f:
    #     f.write(f"[Mirheo run] a={a}, gamma={gamma}, power={power}, sig={sig}, rho_s = {Xi}\n")

    # Run the simulation
    run_Poiseuille(p=p, ranks=(1,1,1), dump=False, comm=comm, out=(folder, name))
    
    # Log the file opening
    # if rank == 0:
    #   with open("logs/korali.log", "a") as f:
    #     f.write(f"[Opening] {folder+name+'prof_*.h5'}\n")

    # Collect the result of the simulation: the velocity profile averaged
    # after the flow has reached a stationary state
    file = glob.glob(folder+name+'prof_*.h5')[0]
    f_in = h5py.File(file)

    # Log the viscosity computation
    # if rank == 0:
    #   with open("logs/korali.log", "a") as f:
    #     f.write(f"[Viscosity computation] {file}\n")

    # Compute the viscosity by fitting the velocity profile to a parabola
    M=np.mean(f_in['velocities'][:,:,:,0], axis=(0,2))
    iL = int(M.shape[0]/2)
    data_neg=M[:iL]
    data_pos=M[iL:]
    data=0.5*(-data_neg+data_pos) # average the two half profiles 

    xmin=0.5
    xmax=iL-0.5
    x=np.linspace(xmin, xmax, iL)

    popt, pcov = curve_fit(quadratic_func, x, data)
    eta=popt[0]

    # Log the resulting viscosity
    # if rank == 0:
    #   with open("logs/korali.log", "a") as f:
    #     f.write(f"[Viscosity] {eta} [{float(np.sqrt(np.diag(pcov))[0])}]\n\n")

    # Save the velocity profile if running standalone
    if standalone:
      out=np.concatenate([[a, gamma, Xi, eta], data])
      with open("velo_prof.csv", "w") as f:
        np.savetxt(f, out.reshape(1, out.shape[0]))

    # Output the result 
    s["Evaluations"] += [eta] # Viscosity in simulation units

    # Clean up the simulation files
    if rank == 0:
      shutil.rmtree('velo/' + name)
      shutil.rmtree('restart/' + name)

def viscosity_analytic(s,X):
    # read parameters from Korali

    # rc, sig = s["Parameters"]
    # gamma = 100
    # power = 0.12

    rc, gamma, sig = s["Parameters"]
    power = 0.12

    kBT_s = constants['kBT_s'] # Energy scale of the DPD particles
    m = constants['m'] # Mass of a DPD bead

    s["Reference Evaluations"] = []
    s["Standard Deviation"] = []

    for rho_i in X:
        visco = kineticVisco(kBT_s, 2.*power, rho_i, rc, gamma, m)
        s["Reference Evaluations"] += [visco] 
        s["Standard Deviation"] += [sig]

def viscosity_analytic_prop(s,X):
    # rc = s['Parameters'][0]
    # gamma = s['Parameters'][1]
    # power = s["Parameters"][2]
    # sig = s["Parameters"][3]

    rc, gamma, sig = s["Parameters"]
    power = 0.12
    
    kBT_s = constants['kBT_s'] # Energy scale of the DPD particles
    m = constants['m'] # Mass of a DPD bead

    s['X'] = X.tolist()
    s['sigma'] = sig
    s['Evaluations'] = []
    for rho_i in X:
        visco = kineticVisco(kBT_s, 2.*power, rho_i, rc, gamma, m)
        s['Evaluations'] += [visco]

############################################################
##################### REFERENCE DATA #######################
############################################################

def getReferencePointsVisco():
  """
  Returns the density in DPD units
  """
  list_rho_s=np.loadtxt('data/data_density_viscosity_DPD.dat', skiprows=1)[:,0]
  return list(list_rho_s)[::2]

def getReferenceDataVisco():
  """
  Returns the viscosity in DPD units
  """
  list_eta_s=np.loadtxt('data/data_density_viscosity_DPD.dat', skiprows=1)[:,1]
  return list(list_eta_s)[::2]

def getReferencePointsSpeed():
  """
  Returns the density in DPD units
  """
  list_rho_s=np.loadtxt('data/data_density_speed_DPD.dat', skiprows=1)[:,0]
  return list(list_rho_s)[::2]

def getReferenceDataSpeed():
  """
  Returns the speed in DPD units
  """
  list_c_s=np.loadtxt('data/data_density_speed_DPD.dat', skiprows=1)[:,1]
  return list(list_c_s)[::2]

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

  
if __name__ == '__main__':
    main(sys.argv[1:]) 