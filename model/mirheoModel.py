#!/usr/bin/env python

import mirheo as mir
from mpi4py import MPI
import sys
import numpy as np
import h5py
from scipy.optimize import curve_fit
import pandas as pd
import os

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

def timeStep(kBT, s, rho_s, rc, gamma, m, a, Fx):
    h = rho_s**(-1/3) # Particle spacing
    nu = kineticVisco(kBT, s, rho_s, rc, gamma, m) # Kinematic viscosity from Lucas' thesis 
    c_s = soundCelerity(kBT, s, rho_s, rc, gamma, m, a) # Sound speed

    dt1 = h**2 /(8*nu) # Viscous diffusion constraint
    dt2 = h/(4*c_s) # Acoustic CFL constraint
    dt3 = 0.25*np.sqrt(h/Fx) # External force constraint

    return(min(dt1, dt2, dt3)/2.)

def get_visco(file, L, Fx):
    f = h5py.File(file)

    # Compute the viscosity by fitting the velocity profile to a parabola
    M=np.mean(f['velocities'][:,:,:,0], axis=(0,2))
    data_neg=M[:L]
    data_pos=M[L:]
    data=0.5*(-data_neg+data_pos) # average the two half profiles 

    xmin=0.5
    xmax=L-0.5
    x=np.linspace(xmin, xmax, L)

    def quadratic_func(y, eta):
        return ((Fx*L)/(2.*eta))*y*(1.-y/L)

    popt, pcov = curve_fit(quadratic_func, x, data)
    eta=popt[0]
    return eta

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
        
    # Compute time step following Lucas' thesis
    dt = timeStep(kBT=kBT, s=2.*power, rho_s=nd, rc=rc, gamma=gamma, m=m, a=a, Fx=Fx)/2.0

    Lx = L 
    Ly = 2*L 
    Lz = 2*L
    domain = (Lx,Ly,Lz)	# domain

    # stslik = 10
    # nsteps = int(tmax/dt)
    # nevery = int(nsteps/stslik)

    runtime = 10
    nsteps_per_runtime = int(runtime/dt)
    
    output_time = 10
    nsteps_per_output = int(output_time/dt)

    #print(f"dt: {dt}, nsteps_per_runtime: {nsteps_per_runtime}, nsteps_per_output: {nsteps_per_output}")    

    # Instantiate Mirheo simulation
    u = mir.Mirheo(nranks=ranks, domain=domain, debug_level=0, 
                   log_filename='log', no_splash=True, comm_ptr=MPI._addressof(comm),
                   checkpoint_every=nsteps_per_runtime-1, 
                   checkpoint_folder='restart/'+name,
                   checkpoint_mode='PingPong'
                   )

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

    #print('+ Equilibrating')
    
    # The plugin createStats does not allow the internal creation of a directory,
    # so we need to create it before running the simulation.
    os.makedirs('stats/'+name, exist_ok = True) 
    f_stats = 'stats/'+name+'stats.csv' # Where the stats of this simulation will be stored
    
    stats_cols = ['time', 'kBT', 'vx', 'vy', 'vz', 'maxv', 'num_particles', 'simulation_time_per_step']
        
    equilibration = True
    if equilibration:
        n_restart = 0
        stats=mir.Plugins.createStats('stats0', every=nsteps_per_output, filename=f_stats)
        u.registerPlugins(stats)
        u.run(nsteps_per_runtime, dt=dt)
        u.deregisterPlugins(stats)

        # Set up the rewritable stats file
        n_restart += 1
        stats=mir.Plugins.createStats(f'stats{n_restart}', every=nsteps_per_output, filename=f_stats)
        u.registerPlugins(stats)
        u.restart(folder = 'restart/'+name)
        u.run(nsteps_per_runtime, dt=dt)
        u.deregisterPlugins(stats)

        # Stop simulation if the average velocity does not increase anymore
        last_velocity = 1
        new_velocity = pd.read_csv(f_stats, names=stats_cols)['maxv'][0]

        while (new_velocity-last_velocity)/last_velocity > 1e-4:
            last_velocity = new_velocity
            n_restart += 1
            
            stats=mir.Plugins.createStats(f'stats{n_restart}', every=nsteps_per_output, filename=f_stats)
            u.registerPlugins(stats)
            
            u.restart(folder = 'restart/'+name)
            u.run(nsteps_per_runtime, dt=dt)
            
            u.deregisterPlugins(stats)
            new_velocity = pd.read_csv(f_stats, names=stats_cols)['maxv'][0]  
           
    # System is in stationary state, now we can sample the velocity profile
    #print('+ Sampling velocity profile')
    n_restart += 1
    t_sampling = 100
    nsteps_sampling = int(t_sampling/dt)
    sample_every = 2 #int(nsteps_sampling/100) # Average 100 times the spatial velocity profile

    # dump_every must include all the timesteps already computed before sampling
    # nsteps_eq = n_restart*nsteps_per_runtime
    # print(f"time for eq: {nsteps_eq*dt}")
    dump_every   = nsteps_sampling -1 # + nsteps_eq # Dump 1 profile averaging the 100 pre-computed spatial profiles

    
    # print(f"run for nsteps_eq: {nsteps_eq}\nnsteps_sampling: {nsteps_sampling}\nsample_every: {sample_every}\ndump_every: {dump_every}") 
    bin_size     = (1.0, 1.0, 1.0)
    #print("folder+name:", folder+name)
    u.registerPlugins(mir.Plugins.createDumpAverage('field', 
                                                     [water], 
                                                     sample_every, 
                                                     dump_every, 
                                                     bin_size, 
                                                     ["velocities"], 
                                                     folder+name+'prof_'))
    # Mirheo seems to append a more or less random number to the output filename. Be careful. 
    
    u.registerPlugins(mir.Plugins.createStats(f'stats{n_restart}', 
                                              every=nsteps_per_output,
                                               filename='stats'))
    
    u.restart(folder = 'restart/'+name)
    u.run(nsteps_sampling, dt=dt)
    


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