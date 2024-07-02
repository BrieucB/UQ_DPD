#!/usr/bin/env python

from mirheoModel import *
import korali
import sys
import numpy as np

from units import *

def F(s,X):
  import h5py
  from mpi4py import MPI
  from scipy.optimize import curve_fit
  import glob
  import shutil

  # read parameters from Korali
  # a     = 0.12
  # gamma = 75.0 
  # power = 0.125 
  # sig   = s["Parameters"][3]

  a     = s["Parameters"][0]
  gamma = s["Parameters"][1]
  power = s["Parameters"][2]
  sig   = s["Parameters"][3]

  # Read the MPI Comm assigned by Korali and feed it to the Mirheo simulation
  # If running on stand alone, use standard MPI communicator
  try:
    comm       = korali.getWorkerMPIComm()
    standalone = False
  except TypeError:
     comm       = MPI.COMM_WORLD
     standalone = True

  rank = comm.Get_rank()
  if rank == 0:
    print(comm)

  L         = constants['L']       # Size of the simulation box in the x-direction
  kBT_s     = constants['kBT_s']  # Energy scale of the DPD particles
  m         = constants['m']      # Mass of a DPD bead
  rc        = constants['rc']     # Cutoff radius
  ptan      = obmd['ptan']        # Tangential pressure

  s["Reference Evaluations"] = []
  s["Standard Deviation"]    = []
  #s["error_fit"]             = []

  n_ref=0
  for Xi in X: # Loop over the reference points: here on density
    
    # Export the simulation parameters
    simu_param = {'m':m, 'nd':Xi, 'rc':rc, 'L':L, 'ptan':ptan}
    dpd_param  = {'a':a, 'gamma':gamma, 'kBT':kBT_s, 'power':power}
    p          = {'simu':simu_param, 'dpd':dpd_param, 'obmd':obmd}

    # Set output file. Mirheo seems to attribute a random number to the output name, making it
    # difficult to find the output file. Here we specify the output folder to retrieve the file.
    folder = "velocities/"
    name   = 'a%.2f_gamma%.2f_power%.2f_n%d/'%(a,gamma,power,n_ref)
    n_ref += 1

    # Log the run
    if rank == 0:
      with open("logs/korali.log", "a") as f:
        f.write(f"[Mirheo run] a={a}, gamma={gamma}, power={power}, sig={sig}, rho_s = {Xi}\n")

    # Run the simulation
    #run_Poiseuille(p=p, ranks=(1,1,1), dump=False, comm=comm, out=(folder, name))
    speed = compute_speed_of_sound(p=p, ranks=(1,1,1), comm=comm, out=(folder, name))
    
    # Output the result 
    s["Reference Evaluations"] += [speed] # Viscosity in simulation units
    s["Standard Deviation"]    += [sig] # Viscosity in simulation units

    # Compute the error and store it: can be accessed after inference to check
    # the quality of the fit
    #s["error_fit"] += [float(np.sqrt(np.diag(pcov))[0])]

    # Clean up the simulation files 
    # -> [TODO] what happens of this on multiple nodes/GPUs ?
    if rank == 0:
      try:
        shutil.rmtree('velo/' + name)
        shutil.rmtree('restart/' + name)
      except:
        #print("Error: could not remove the folder velo/ or restart/")
        pass
      

def getReferencePoints():
  """
  Returns the density in DPD units
  """
  list_rho_s=np.loadtxt('data/data_density_speed_DPD.dat', skiprows=1)[:,0]
  return list(list_rho_s)[0:1] #[::2]

def getReferenceData():
  """
  Returns the viscosity in DPD units
  """
  list_eta_s=np.loadtxt('data/data_density_speed_DPD.dat', skiprows=1)[:,1]
  return list(list_eta_s)[0:1] #[::2]

def main(argv):
  import argparse
  import shutil
  from mpi4py import MPI

  try:
    shutil.rmtree('./velocities')
    shutil.rmtree('./velo')
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

  X = getReferencePoints()
  #print(getReferenceData())

  F(s,X)

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  if rank == 0:
    print("rho:", X, "s[\"Reference Evaluations\"]:", s["Reference Evaluations"], "s[\"error_fit\"]:", 100*np.array(s["error_fit"])/np.array(s["Reference Evaluations"]), "%")
    print("s[\"Standard Deviation\"]:", s["Standard Deviation"][0])

if __name__ == '__main__':
    main(sys.argv[1:]) 