#!/usr/bin/env python

import mirheoOBMD as mir
from mpi4py import MPI
import sys
import numpy as np
import h5py
from scipy.optimize import curve_fit
import time


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

def compute_density(p: dict,
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
    m           = p['simu']['m']
    nd          = p['simu']['nd']
    rc          = p['simu']['rc']
    L           = p['simu']['L']

    # Collect DPD parameters
    a           = p['dpd']['a']
    gamma       = p['dpd']['gamma']
    kBT         = p['dpd']['kBT']
    power       = p['dpd']['power']

    # Collect OBMD parameters
    bufferSize  = p['obmd']['bufferSize']
    bufferAlpha = p['obmd']['bufferAlpha']
    bufferTau   = p['obmd']['bufferTau']
    ptan        = 0 #p['obmd']['ptan']
    pext        = p['obmd']['pext']

    # OBMD parameters
    #alpha=0.103
    #pext = nd*kBT + alpha*a*nd**2

    # Set output path
    folder, name = out[0], out[1]
    rank = comm.Get_rank()

    # Output parameters on the current parameter set
    if rank == 0:
        print(p)
        
    # Set domain size
    Lx = L 
    Ly = L 
    Lz = L
    domain = (Lx,Ly,Lz)	# domain

    # Compute time step following Lucas' thesis
    dt = timeStep(kBT=kBT, s=2.*power, rho_s=nd, rc=rc, gamma=gamma, m=m, a=a, Fx=pext)
    #dt = 0.001
    #print('dt =', dt)

    # Set runtime and output time for adaptative equilibration
    runtime = 1
    nsteps_per_runtime = int(runtime/dt)
    
    output_time = runtime
    nsteps_per_output = int(output_time/dt)

    t_eq = 20    

    obmd = {
        "bufferSize"   : bufferSize*Lx,
        "bufferAlpha"  : bufferAlpha,
        "bufferTau"    : bufferTau*dt,
        "pext"         : pext,
        "ptan"         : ptan
    }

    # Instantiate Mirheo simulation
    u = mir.Mirheo(nranks            = ranks,
                   domain            = domain,
                   debug_level       = 0,
                   log_filename      = 'logs/log',
                   no_splash         = True,
                   comm_ptr          = MPI._addressof(comm),
                   checkpoint_every  = int(t_eq/dt) -1, #nsteps_per_runtime-1
                   checkpoint_folder = 'restart/'+name,
                   checkpoint_mode   = 'PingPong', 
                   **obmd
                   )

    water    = mir.ParticleVectors.ParticleVector('water', mass = m, obmd = 1)
    ic_water = mir.InitialConditions.Uniform(number_density = nd)
    u.registerParticleVector(water, ic_water)      # Register the PV and initialize its particles

    # Create and register DPD interaction with specific parameters and cutoff radius
    dpd_wat = mir.Interactions.Pairwise(name  = 'dpd_wat', 
                                        rc    = rc, 
                                        kind  = "DPD", 
                                        a     = a, 
                                        gamma = gamma, 
                                        kBT   = kBT, 
                                        power = power)
    u.registerInteraction(dpd_wat)
    u.setInteraction(dpd_wat, water, water)

    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)
    u.setIntegrator(vv, water)
        
    adaptative_equilibration = False
    n_restart     = 0
    bin_size      = (1., Ly, Lz)#(1.0, 1.0, 1.0)
    

    if adaptative_equilibration:
        f_velo = 'velo/' + name + 'prof_' + str(n_restart)
        #print(f_velo)
        veloField = mir.Plugins.createDumpAverage(name         = f'field{n_restart}', 
                                                  pvs          = [water], 
                                                  sample_every = 2, 
                                                  dump_every   = nsteps_per_output-1, 
                                                  bin_size     = bin_size, 
                                                  channels     = ["velocities"], 
                                                  path         = f_velo)
        u.registerPlugins(veloField)
        u.run(nsteps_per_runtime, dt=dt)
        u.deregisterPlugins(veloField)
                
        # Stop simulation if the viscosity stabilizes
        win                  = 100
        list_visco           = [1000 for i in range(win)]
        indx                 = 0
        new_visco            = get_visco(f_velo+'00000.h5', L, nd)
        list_visco[indx%win] = new_visco
        indx                += 1

        #while np.abs(new_visco-last_visco) > 1e-3:
        while (np.std(list_visco)/np.mean(list_visco) > 1e-2) and (n_restart < 1000):
            #last_visco = new_visco
            start_time = time.time()
            n_restart += 1            
            #f_velo     = 'velo/'+name+'prof_'+str(n_restart)
            #print(f_velo)

            veloField = mir.Plugins.createDumpAverage(name         = f'field{n_restart}', 
                                                      pvs          = [water], 
                                                      sample_every = 2, 
                                                      dump_every   = nsteps_per_output-1, 
                                                      bin_size     = bin_size, 
                                                      channels     = ["velocities"], 
                                                      path         = f_velo)
            u.registerPlugins(veloField)
            u.restart(folder = 'restart/'+name)
            u.run(nsteps_per_runtime, dt=dt)
            u.deregisterPlugins(veloField)
            print("--- %s seconds by run ---" % (time.time() - start_time))

            start_time = time.time()
            new_visco = get_visco(f_velo+'%05d.h5'%n_restart, L, nd)
            list_visco[indx%win] = new_visco
            indx += 1
            print("--- %s seconds to compute visco ---" % (time.time() - start_time))
            #print('new_visco', new_visco)
            if rank == 0:
                with open("logs/mirheo.log", "a") as f:
                    f.write(f'[Mirheo run] mean_visco = {np.mean(list_visco)}, std/mean = {np.std(list_visco)/np.mean(list_visco)}, {n_restart}\n')
            print('mean_visco =', np.mean(list_visco), 'std/mean =', np.std(list_visco)/np.mean(list_visco), n_restart)
    else:
        u.run(int(t_eq/dt), dt=dt)

    # System is in stationary state, now we can sample the velocity profile
    n_restart      += 1
    t_sampling      = 50
    nsteps_sampling = int(t_sampling/dt)
    sample_every    = 2 
    dump_every      = nsteps_sampling -1 

    #print(folder+name+'prof_')
    densField = mir.Plugins.createDumpAverage(       'field', 
                                                     [water], 
                                                     sample_every, 
                                                     dump_every, 
                                                     bin_size, 
                                                     ["velocities"], # interested only in density, but need to also output this for the plugin to work
                                                     folder+name+'prof_pext%.2f_'%pext)
    u.registerPlugins(densField)
    # Mirheo seems to append a more or less random number to the output filename. Be careful. 

    u.restart(folder = 'restart/'+name)
    u.run(nsteps_sampling, dt=dt)
    u.deregisterPlugins(densField)
    
    del u

def compute_speed_of_sound(p: dict,
                   comm: MPI.Comm,
                   out: tuple,
                   ranks: tuple=(1,1,1)):
    
    import glob
    import h5py
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt
    #from matplotlib.pyplot import plot, savefig, clf
    folder, name = out[0], out[1]

    pext = p['simu']['nd']*p['dpd']['kBT'] + 0.103*p['dpd']['a']*p['simu']['nd']**2
    list_pext=np.linspace(0.1*pext, 10.*pext, 8)

    list_density = []

    for pext in list_pext:
        p['obmd']['pext']=pext
        #compute_density(p, comm, out, ranks)

        #print(folder+name+'prof_pext%.2f*.h5'%pext)
        file = glob.glob(folder+name+'prof_pext%.2f*.h5'%pext)[0]
        f_in = h5py.File(file)

        rhox = f_in['number_densities'][0,0,:]

        #print(np.mean(rhox[2:-2]))
        list_density.append(np.mean(rhox[4:-4]))

    def func(rho, a, b): # p(rho)
        # quadratic equation of state
        return a * rho * rho + b * rho
    
    popt_open, pcov_open = curve_fit(func, list_density, list_pext)
    c_sound_open = np.sqrt(2*popt_open[0]*p['simu']['nd'] + popt_open[1])
    print('speed of sound: ', c_sound_open)
    #plt.figure(figsize=(4, 3))
    #error_open = np.sqrt(np.diag(pcov_open))
    #error_c_open = np.sqrt(np.sqrt(2*n*2*n*error_open[0]+error_open[1]))

    plt.clf()
    plt.plot(list_density, list_pext, 'o', label='data')
    vec_density = np.linspace(0.99*min(list_density), 1.01*max(list_density), 100)
    plt.plot(vec_density, func(vec_density, *popt_open), '-', label='fit')
    plt.savefig(folder+name+'eos.png')

    print('analytical', soundCelerity(1, 0.25, 3, 1, 4.5, 1, 200))

    return(c_sound_open)

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