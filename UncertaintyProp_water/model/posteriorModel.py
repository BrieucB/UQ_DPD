#!/usr/bin/env python

import korali
import sys
sys.path.append('./model')
import numpy as np

def kineticVisco(kBT, s, rho_s, rc, gamma, m):
    """
    s = power*2
    """
    A=(3*kBT*(s+1)*(s+2)*(s+3))/(16*np.pi*rho_s*(rc**3)*gamma)
    B=(16*np.pi*rho_s*(rc**3)*gamma)/(5*m*(s+1)*(s+2)*(s+3)*(s+4)*(s+5))
    return((A+B)/rho_s)

def viscosity_analytic(s,X):
  
    # read parameters from Korali
    # rc = s["Parameters"][0]
    # gamma = s["Parameters"][1]
    # power = s["Parameters"][2]
    # sig = s["Parameters"][3]

    rc = s["Parameters"][0]
    gamma = 100
    power = 0.12
    sig = s["Parameters"][1]

    m = 1
    T25 = 25
    import numpy as np
    params=np.loadtxt('metaparam.dat', skiprows=1) # L, Fx, rho_s, kBT_s, pop_size
    rho_s =  params[2]
    kBT_s = params[3]

    s["Reference Evaluations"] = []
    s["Standard Deviation"] = []

    for Ti in X:
        visco = kineticVisco(kBT_s*(Ti+273.15)/(T25+273.15), 2.*power, rho_s, rc, gamma, m)
        s["Reference Evaluations"] += [visco] 
        s["Standard Deviation"] += [sig]

def viscosity_analytic_prop(s,X):
    # rc = s['Parameters'][0]
    # gamma = s['Parameters'][1]
    # power = s["Parameters"][2]
    # sig = s["Parameters"][3]

    rc = s['Parameters'][0]
    gamma = 100
    power = 0.12
    
    sig = s["Parameters"][1]
    s['sigma'] = sig

    m = 1
    T25 = 25

    params=np.loadtxt('metaparam.dat', skiprows=1) # L, Fx, rho_s, kBT_s, pop_size
    rho_s =  params[2]
    kBT_s = params[3]

    s['X'] = X.tolist()
    s['Evaluations'] = []
    for Ti in X:
        visco = kineticVisco(kBT_s*(Ti+273.15)/(T25+273.15), 2.*power, rho_s, rc, gamma, m)
        s['Evaluations'] += [visco]

############################################################
##################### REFERENCE DATA #######################
############################################################

def getReferencePointsVisco():
    """
    Returns the temperature
    """
    import numpy as np
    T = np.loadtxt('data_viscosity.dat', skiprows=1)[:,0] #[1:3,0] # Temperature in °C

    return  list(T)

def getReferenceDataVisco():
  """
  Returns the viscosity in DPD units
  """

  import numpy as np
  params=np.loadtxt('metaparam.dat', skiprows=1) # L, Fx, rho_s, kBT_s, pop_size
  rho_s = params[2]
  kBT_s = params[3]
  
  visco = np.loadtxt('data_viscosity.dat', skiprows=1)[:,1]

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

  # Turn the real data into simulation units
  return list(visco/u_eta)


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