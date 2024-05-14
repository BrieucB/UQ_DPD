#!/usr/bin/env python

import korali
import sys
sys.path.append('./model')
import numpy as np

from units import *

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