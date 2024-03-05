#!/usr/bin/env python

from copy import deepcopy
import mirheo as mir
from mpi4py import MPI
import numpy as np
import sys

def run_shear_flow(*,
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
    shear_rate = p['simu']['shear_rate']

    # Collect DPD parameters
    a=p['dpd']['a']
    gamma=p['dpd']['gamma']
    kBT=p['dpd']['kBT']
    power=p['dpd']['power']

    # Set output path
    folder, name = out[0], out[1]

    print(p)
    #print(folder+ name)

    dt=0.001
    
    # Instantiate Mirheo simulation
    domain = (L, L + 4*rc, L)
    u = mir.Mirheo(ranks, domain, debug_level=0, log_filename='log', no_splash=True, comm_ptr=MPI._addressof(comm))

    pv = mir.ParticleVectors.ParticleVector('pv', mass = m) # Create a simple Particle Vector (PV) named 'pv'
    ic = mir.InitialConditions.Uniform(nd)        # Specify uniform random initial conditions
    u.registerParticleVector(pv, ic)                          # Register the PV and initialize its particles

    # Create and register DPD interaction with specific parameters and cutoff radius
    dpd = mir.Interactions.Pairwise('dpd', rc=rc, kind="DPD", a=a, gamma=gamma, kBT=kBT, power=power, stress=True, stress_period=1)
    u.registerInteraction(dpd)

    # Tell the simulation that the particles of pv interact with dpd interaction
    u.setInteraction(dpd, pv, pv)

    # Create and register Velocity-Verlet integrator
    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)

    # Walls
    vw = shear_rate * L / 2

    plate_lo = mir.Walls.MovingPlane("plate_lo", normal=(0, -1, 0), pointThrough=(0,     2 * rc, 0), velocity=(-vw, 0, 0))
    plate_hi = mir.Walls.MovingPlane("plate_hi", normal=(0,  1, 0), pointThrough=(0, L + 2 * rc, 0), velocity=( vw, 0, 0))

    u.registerWall(plate_lo)
    u.registerWall(plate_hi)

    nequil = int(1.0 / dt)
    frozen_lo = u.makeFrozenWallParticles("plate_lo", walls=[plate_lo], interactions=[dpd], integrator=vv, number_density=nd, nsteps=nequil, dt=dt)
    frozen_hi = u.makeFrozenWallParticles("plate_hi", walls=[plate_hi], interactions=[dpd], integrator=vv, number_density=nd, nsteps=nequil, dt=dt)

    move_lo = mir.Integrators.Translate('move_lo', velocity=(-vw, 0, 0))
    move_hi = mir.Integrators.Translate('move_hi', velocity=( vw, 0, 0))
    u.registerIntegrator(move_lo)
    u.registerIntegrator(move_hi)


    # Set interactions between pvs
    u.setInteraction(dpd, pv, frozen_lo)
    u.setInteraction(dpd, pv, frozen_hi)

    # Set integrators
    u.setIntegrator(vv, pv)
    u.setIntegrator(move_lo, frozen_lo)
    u.setIntegrator(move_hi, frozen_hi)

    u.setWall(plate_lo, pv)
    u.setWall(plate_hi, pv)

    # Dump plugins
    if dump:
        t_dump_every = p['t_dump_every']
        dump_every = int(t_dump_every / dt)
        path = 'ply/'
        #u.registerPlugins(mir.Plugins.createDumpMesh("mesh_dump", pv_rbc, dump_every, path))
        #u.registerPlugins(mir.Plugins.createStats('stats', every=dump_every, filename="stats.csv"))

        # Set the dumping parameters
        sample_every = 2
        dump_every = 1000
        bin_size = (1., 1., 1.)

        # Create and register XDMF plugin
        u.registerPlugins(mir.Plugins.createDumpAverage('field', 
                                                        [pv], 
                                                        sample_every, 
                                                        dump_every, 
                                                        bin_size, ["velocities"], 'h5/solvent-'))

    
        
    omega_sphere = 0.5 * shear_rate
    
    f_sphere = omega_sphere / (2 * np.pi)
    # tend so that a sphere would make a given number of revolutions
    tend = 5 / f_sphere
    #tend = 2 / f_sphere
    nsteps = int(tend / dt)

    sample_every = 1
    dump_every = nsteps-1
    bin_size = (rc, rc, rc)
    u.registerPlugins(mir.Plugins.createDumpAverage('stresses', 
                                                    [pv], 
                                                    sample_every, 
                                                    dump_every, 
                                                    bin_size, ["stresses"], #, "velocities" 
                                                    folder+name))

    
    if u.isMasterTask():
        print(f"Domain = {domain}")
        substeps=1
        print(f"dt = {dt}")
        if dump:
            print(f"t_dump_every = {t_dump_every}")
        print(f"will run for {nsteps} steps ({tend} simulation time units)")
        sys.stdout.flush()

    u.run(nsteps, dt)

def load_parameters(filename: str):
    import pickle

    with open(filename, 'rb') as f:
        return pickle.load(f)
    
def main(argv):
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('parameters', type=str, help="The file containing the parameters of the simulation.")
    parser.add_argument('--dump', action='store_true', default=False, help="Will dump ply files if set to True.")
    parser.add_argument('--ranks', type=int, nargs=3, default=[1,1,1], help="Number of ranks in each direction.")
    args = parser.parse_args(argv)

    p = load_parameters(args.parameters)

    comm = MPI.COMM_WORLD

    run_shear_flow(p=p,
                   ranks=args.ranks,
                   comm=comm,
                   out=[os.getcwd(),"/h5/"],
                   dump=args.dump)


if __name__ == '__main__':
    main(sys.argv[1:])