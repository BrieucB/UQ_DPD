#!/usr/bin/env python

import mirheo as mir
from mpi4py import MPI
import sys
import numpy as np

def kineticVisco(kBT, s, rho_s, rc, gamma, m):
    """
    s = power*2
    """
    A=(3*kBT*(s+1)*(s+2)*(s+3))/(16*np.pi*rho_s*(rc**3)*gamma)
    B=(16*np.pi*rho_s*(rc**3)*gamma)/(5*m*(s+1)*(s+2)*(s+3)*(s+4)*(s+5))
    return(A+B)

def soundCelerity(kBT, s, rho_s, rc, gamma, m, a):
    alpha=0.103
    return(np.sqrt(kBT/m + 2*alpha*rc**4*a*rho_s/m**2))

def timeStep(kBT, s, rho_s, rc, gamma, m, a):
    h = rho_s**(-1/3) # Particle spacing
    nu = kineticVisco(kBT, s, rho_s, rc, gamma, m) # Kinematic viscosity from Lucas' thesis 
    c_s = soundCelerity(kBT, s, rho_s, rc, gamma, m, a) # Sound speed

    dt1 = h**2 /(8*nu) # Viscous diffusion constraint
    dt2 = h/(4*c_s) # Acoustic CFL constraint

    return(min(dt1, dt2)/2.)

def run_Isothermal(*,
                   p: dict,
                   comm: MPI.Comm,
                   out: tuple,
                   ranks: tuple=(1,1,1),
                   dump: bool=False):
    """
    Argument:
        p: parameters of the simulation and DPD parameters.
        comm: each Mirheo simulation needs its own MPI_COMM assigned by Korali.
        out: folder + name of the output file.
        ranks: Mirheo ranks
        dump: if True, will dump simu data over time (TODO).
    """

    # Collect parameters of the simulation 
    m    = p['simu']['m']
    nd   = p['simu']['nd']
    rc   = p['simu']['rc']
    L    = p['simu']['L']
    tmax = 200 #p['simu']['tmax']

    # Collect DPD parameters
    a     = p['dpd']['a']
    gamma = p['dpd']['gamma']
    kBT   = p['dpd']['kBT']
    power = p['dpd']['power']

    # Set output path
    folder, name = out[0], out[1]

    rank = comm.Get_rank()
    if rank == 0:
        print('[Korali] Compressibility computation\n', p)
        
    # Compute time step following Lucas' thesis
    dt = timeStep(kBT=kBT, s=2.*power, rho_s=nd, rc=rc, gamma=gamma, m=m, a=a) 

    Lx = L  
    Ly = L 
    Lz = L
    domain = (Lx,Ly,Lz)	# domain

    stslik = 100
    nsteps = int(tmax/dt)
    nevery = int(nsteps/stslik)
    
    # Instantiate Mirheo simulation
    u = mir.Mirheo(ranks, domain, debug_level=0, 
                   log_filename='log', no_splash=True, comm_ptr=MPI._addressof(comm))

    water = mir.ParticleVectors.ParticleVector('water', mass = m)
    ic_water = mir.InitialConditions.Uniform(number_density = nd)
    u.registerParticleVector(water, ic_water)      # Register the PV and initialize its particles

    #print(water.getCoordinates())

    # Create and register DPD interaction with specific parameters and cutoff radius
    dpd_wat = mir.Interactions.Pairwise('dpd_wat', 
                                        rc     = rc, 
                                        kind   = "DPD", 
                                        a      = a, 
                                        gamma  = gamma, 
                                        kBT    = kBT, 
                                        power  = power,
                                        stress = True,
                                        stress_period = dt)
    
    u.registerInteraction(dpd_wat)
    u.setInteraction(dpd_wat, water, water)

    vv = mir.Integrators.VelocityVerlet('vv')

    u.registerIntegrator(vv)
    u.setIntegrator(vv, water)

    dump_every   = nevery 
    bin_size     = (1.0, 1.0, 1.0)

    # Output virial stress
    virial = mir.Plugins.createVirialPressurePlugin(name='dpd_wat', 
                                                    pv=water, 
                                                    regionFunc=lambda u : 1, 
                                                    h=bin_size, 
                                                    dump_every=1, 
                                                    path=folder+name)
    u.registerPlugins(virial)

    # Monitor the simulation to check the temperature
    #stats=mir.Plugins.createStats('stats0', every=dump_every)
    #u.registerPlugins(stats)

    u.run(nsteps, dt=dt)

def getPressure(file, N, kBT, Lx, Ly, Lz):
    pres = np.loadtxt(file, delimiter=',', skiprows=1)
    return(N*kBT/(Lx*Ly*Lz) + np.mean(pres[-5000:,1])/(Lx*Ly*Lz))
    
def getCompressibility(p: dict,
                       comm: MPI.Comm,
                       out: tuple,
                       ranks: tuple=(1,1,1)):

    L = p['simu']['L']
    V = L**3
    folder, name = out[0], out[1]

    # Derivative is computed using central finite difference at order 1
    list_L = np.array([0.999*L, 1.001*L])
    list_V = list_L**3
    dV = list_V[1]-list_V[0]

    N=int(p['simu']['nd']*L**3)

    # Estimate compressibility
    P=[]
    for L in list_L:
        Lx = Ly = Lz = L
        p['simu']['nd'] = N/(Lx*Ly*Lz)
        p['simu']['L'] = L
        run_Isothermal(p=p,
                    ranks=ranks,
                    comm=comm,
                    out=out,
                    dump=False)
        
        kBT  = p['dpd']['kBT']
        Pi = getPressure(folder+name+'water.csv', N, kBT, Lx, Ly, Lz)
        P.append(Pi)

    #print(P,V, dV)
    dP = P[1]-P[0]
    return(-(1/V)*dV/dP)

def load_parameters(filename: str):
    import pickle

    with open(filename, 'rb') as f:
        return pickle.load(f)
    
def main(argv):
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('parameters', type=str, 
                        help="The file containing the parameters of the simulation.")
    parser.add_argument('--dump', action='store_true', default=False, 
                        help="Will dump ply files if set to True.")
    parser.add_argument('--ranks', type=int, nargs=3, default=[1,1,1], 
                        help="Number of ranks in each direction.")
    args = parser.parse_args(argv)

    p = load_parameters(args.parameters)

    comm = MPI.COMM_WORLD
    
    getCompressibility(p=p,
                       comm=comm,
                       out='./compressibility')


if __name__ == '__main__':
    main(sys.argv[1:]) 