#!/usr/bin/env python

from mirheoModel import *
import korali
import sys
import numpy as np

#from units import *

import trimesh
import yaml

def compute_pbuckling(sample,X):
  import h5py
  from mpi4py import MPI
  from scipy.optimize import curve_fit
  import glob
  import shutil

  # Read general parameters
  filename_default = 'model/parameter/parameters-default.yaml'
  with open(filename_default, 'rb') as f:
      parameters_default = yaml.load(f, Loader = yaml.CLoader)
      
  filename = 'model/parameter/parameters.yaml'
  with open(filename, 'rb') as f:
      parameters = yaml.load(f, Loader = yaml.CLoader)    

  filename_prms = 'model/parameter/parameters.prms.yaml'
  with open(filename_prms, 'rb') as f:
      prms_emb = yaml.load(f, Loader = yaml.CLoader)

  pos_q = np.reshape(np.loadtxt('model/posq.txt'), (-1, 7))

  p = {**parameters_default, **parameters, 'emb':prms_emb, 'pos_q': pos_q}

  # Read parameters to be optimized from Korali
  ka, sig = sample["Parameters"]

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

    # Set output file
    folder = "out/"
    name   = 'ka%.2f_n%d/'%(ka,n_ref)
    n_ref += 1

    # Run the simulation
    p['emb']['ka'] = ka
    pbuckling = mirheo_pbuckling(p=p, ranks=(1,1,1), comm=comm, out=(folder, name), dump=False)
    print(pbuckling)
    
    # Output the result 
    sample["Reference Evaluations"] += [pbuckling] 
    sample["Standard Deviation"]    += [sig]    

def getReferencePoints():
  """
  Returns the density in DPD units
  """
  list_rho_s=np.loadtxt('data/data_X_pbuckling_DPD.dat', skiprows=1).reshape(1,-1)[:,0]
  return list(list_rho_s)

def getReferenceData():
  """
  Returns the viscosity in DPD units
  """
  list_pb_s=np.loadtxt('data/data_X_pbuckling_DPD.dat', skiprows=1).reshape(1,-1)[:,1]
  return list(list_pb_s)

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

  #F(s,X)

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  if rank == 0:
    print("rho:", X, "s[\"Reference Evaluations\"]:", s["Reference Evaluations"], "s[\"error_fit\"]:", 100*np.array(s["error_fit"])/np.array(s["Reference Evaluations"]), "%")
    print("s[\"Standard Deviation\"]:", s["Standard Deviation"][0])

if __name__ == '__main__':
    main(sys.argv[1:]) 