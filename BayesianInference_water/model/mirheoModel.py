#!/usr/bin/env python

import mirheo as mir
from mpi4py import MPI
import sys
import numpy as np
import h5py
from scipy.optimize import curve_fit

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

def get_visco(file, L, Fx, rho):
    f = h5py.File(file)
    L=int(L)

    # Compute the viscosity by fitting the velocity profile to a parabola
    M=np.mean(f['velocities'][:,:,:,0], axis=(0,2))
    data_neg=M[:L]
    data_pos=M[L:]
    data=0.5*(-data_neg+data_pos) # average the two half profiles 

    xmin=0.5
    xmax=L-0.5
    x=np.linspace(xmin, xmax, L)

    def quadratic_func(y, eta):
        return ((rho*Fx*L)/(2.*eta))*y*(1.-y/L)

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
    dt = timeStep(kBT=kBT, s=2.*power, rho_s=nd, rc=rc, gamma=gamma, m=m, a=a, Fx=Fx)

    Lx = L 
    Ly = 2*L 
    Lz = 2*L
    domain = (Lx,Ly,Lz)	# domain

    runtime = 1
    nsteps_per_runtime = int(runtime/dt)
    
    output_time = 1
    nsteps_per_output = int(output_time/dt)

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

    u.registerIntegrator(vv)
    u.setIntegrator(vv, water)
        
    equilibration = True
    n_restart = 0
    bin_size     = (1.0, 1.0, 1.0)

    if equilibration:
        f_velo = 'velo/' + name + 'prof_' + str(n_restart)
        veloField = mir.Plugins.createDumpAverage(f'field{n_restart}', 
                                                    [water], 
                                                    2, 
                                                    nsteps_per_output-1, 
                                                    bin_size, 
                                                    ["velocities"], 
                                                    f_velo)
        u.registerPlugins(veloField)
        u.run(nsteps_per_runtime, dt=dt)
        u.deregisterPlugins(veloField)

        # Stop simulation if the viscosity stabilizes
        last_visco = 100        
        new_visco = get_visco(f_velo+'00000.h5', L, Fx, nd)

        while np.abs(new_visco-last_visco) > 1e-3:
            last_visco = new_visco
            n_restart += 1
            
            f_velo = 'velo/'+name+'prof_'+str(n_restart)
            veloField = mir.Plugins.createDumpAverage(f'field{n_restart}', 
                                                    [water], 
                                                    2, 
                                                    nsteps_per_output-1, 
                                                    bin_size, 
                                                    ["velocities"], 
                                                    f_velo)
            u.registerPlugins(veloField)
            u.restart(folder = 'restart/'+name)
            u.run(nsteps_per_runtime, dt=dt)
            u.deregisterPlugins(veloField)

            new_visco = get_visco(f_velo+'%05d.h5'%n_restart, L, Fx, nd)
            print('new_visco', new_visco)
           
    # System is in stationary state, now we can sample the velocity profile
    n_restart += 1
    t_sampling = 50
    nsteps_sampling = int(t_sampling/dt)
    sample_every = 2 
    dump_every   = nsteps_sampling -1 

    u.registerPlugins(mir.Plugins.createDumpAverage('field', 
                                                     [water], 
                                                     sample_every, 
                                                     dump_every, 
                                                     bin_size, 
                                                     ["velocities"], 
                                                     folder+name+'prof_'))
    # Mirheo seems to append a more or less random number to the output filename. Be careful. 

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