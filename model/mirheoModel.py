#!/usr/bin/env python

from copy import deepcopy
import mirheo as mir
from mpi4py import MPI
import numpy as np
import sys

def run_Poiseuille(*,
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
    m = p['simu']['m']
    nd = p['simu']['nd']
    rc = p['simu']['rc']
    L = p['simu']['L']
    Fx = p['simu']['Fx']
    tmax = p['simu']['tmax']

    # Collect DPD parameters
    a=p['dpd']['a']
    gamma=p['dpd']['gamma']
    kBT=p['dpd']['kBT']
    power=p['dpd']['power']

    # Set output path
    folder, name = out[0], out[1]

    rank = comm.Get_rank()

    if rank == 0:
        print(p)
        
    #print(folder+ name)

    dt=0.001
    
    Lx = L #16.0 #32.0
    Ly = 2*L #32.0 #64.0
    Lz = 2*L #32.0 #64.0
    domain = (Lx,Ly,Lz)	# domain

    D = 0.01
    T_diff = L**2 / (2*D) # where D propto 1/viscosity

    stslik = 10
    nsteps = int(tmax/dt) #400000 #int(T_diff/dt) # find a way to stop the simulation when the flow is stabilized
    print(nsteps)
    nevery = int(nsteps/stslik)
    
    # Instantiate Mirheo simulation
    u = mir.Mirheo(ranks, domain, debug_level=0, 
                   log_filename='log', no_splash=True, comm_ptr=MPI._addressof(comm))

    water = mir.ParticleVectors.ParticleVector('water', mass = m)
    ic_water = mir.InitialConditions.Uniform(number_density = nd)
    u.registerParticleVector(water, ic_water)      # Register the PV and initialize its particles

    # Create and register DPD interaction with specific parameters and cutoff radius
    dpd_wat = mir.Interactions.Pairwise('dpd_wat', 
                                        rc=rc, 
                                        kind="DPD", 
                                        a=a, 
                                        gamma=gamma, 
                                        kBT=kBT, 
                                        power=power)
    u.registerInteraction(dpd_wat)
    u.setInteraction(dpd_wat, water, water)

    vv = mir.Integrators.VelocityVerlet_withPeriodicForce('vv', force=Fx, direction='x')
    # Compute momentum conservation? and compare with eq case.

    u.registerIntegrator(vv)
    u.setIntegrator(vv, water)

    sample_every = nevery
    dump_every   = nevery #int((nsteps-1)/2)
    bin_size     = (1.0, 1.0, 1.0)

    #print(sample_every, dump_every, folder+name)

    u.registerPlugins(mir.Plugins.createDumpAverage('field', 
                                                    [water], 
                                                    sample_every, 
                                                    dump_every, 
                                                    bin_size, 
                                                    ["velocities"], 
                                                    folder+name))
    

    u.run(nsteps, dt=dt)
    

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

    run_Poiseuille(p=p,
                   ranks=args.ranks,
                   comm=comm,
                   out=[os.getcwd(),"/h5/"],
                   dump=args.dump)


if __name__ == '__main__':
    main(sys.argv[1:])