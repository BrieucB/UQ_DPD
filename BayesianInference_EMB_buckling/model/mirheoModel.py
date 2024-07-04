#!/usr/bin/env python

import mirheo as mir
from mpi4py import MPI
import sys
import numpy as np
import h5py
from scipy.optimize import curve_fit
import time

import trimesh

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


def run_equil_EMB(*,
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

    # Set output path
    folder, name = out[0], out[1]
    rank = comm.Get_rank()

    # Output parameters on the current parameter set
    if rank == 0:
        print(p)

    # Compute time step following Lucas' thesis
    #dt = timeStep(kBT=kBT, s=2.*power, rho_s=nd, rc=rc, gamma=gamma, m=m, a=a, Fx=ptan)
    #dt = min(dt, 0.5e-2)
    dt = 0.001
    
    ######################################################
    # set-up parameters

    objFile = p["objFile"]
    numsteps = p["numsteps"]
    dt = p["dt"]
    Lx = p["Lx"]
    Ly = p["Ly"]
    Lz = p["Lz"]
    nevery = p["nevery"]
    rhow = p["rhow"]
    rhog = p["rhog"]
    alpha = p["alpha"]
    aii = p["aii"]
    strong = p["strong"]
    gamma_dpd = p["gamma_dpd"]
    rc = p["rc"]
    s = p["s"] 
    kbt = p["kbt"]
    obmd_flag = p["obmd_flag"]
    mvert = p["mvert"]
    mw = p["mw"]
    mg = p["mg"]

    pos_q = np.reshape(np.loadtxt('posq.txt'), (-1, 7))

    ranks = (1, 1, 1)                       
    domain = (Lx, Ly, Lz)   

    ######################################################
    checkpoint_step = numsteps - 1

    #mirheo coordinator
    u = mir.Mirheo(ranks, domain, debug_level = 3, log_filename = 'logs/log', checkpoint_folder = "restart/", checkpoint_every = checkpoint_step)

    #loads the off script
    mesh = trimesh.load_mesh(objFile)

    #reads vertices, faces
    mesh_emb = mir.ParticleVectors.MembraneMesh(mesh.vertices.tolist(), mesh.faces.tolist())

    emb = mir.ParticleVectors.MembraneVector("emb", mass = mvert, mesh = mesh_emb)

    #initial condition for EMB
    ic_emb   = mir.InitialConditions.Membrane(pos_q)

    #register
    u.registerParticleVector(emb, ic_emb)

    #water
    water = mir.ParticleVectors.ParticleVector('water', mass = mw)#, obmd = obmd_flag) # ne smes imeti istega imena "water" za več "pv"-jev, "water" je ime "pv"-ja znotraj mirhea
    ic_water = mir.InitialConditions.Uniform(number_density = rhow)
    u.registerParticleVector(water, ic_water)

    #solvent 
    sol2 = mir.ParticleVectors.ParticleVector('sol2', mass = mg)#, obmd = obmd_flag)
    ic_outer2 = mir.InitialConditions.Uniform(number_density = rhog)
    u.registerParticleVector(sol2, ic_outer2)

    #splits the particle vector into inner part and outer part defined by the membrane
    #only one can be not null, either inside or outside
    inner_checker_1 = mir.BelongingCheckers.Mesh("inner_checker_1")
    u.registerObjectBelongingChecker(inner_checker_1, emb)
    gas = u.applyObjectBelongingChecker(inner_checker_1, sol2, correct_every = 0, inside = "gas", outside = "") 
    #https://mirheo.readthedocs.io/en/latest/user/tutorials.html

    inner_checker_2 = mir.BelongingCheckers.Mesh("inner_solvent_checker_2")
    u.registerObjectBelongingChecker(inner_checker_2, emb)
    u.applyObjectBelongingChecker(inner_checker_2, water, correct_every = 0, inside = "none", outside = "")

    #interactions
    int_emb = mir.Interactions.MembraneForces("int_emb", "Lim", "KantorStressFree", **prms_emb, stress_free = True) 
    dpd_thermostat = mir.Interactions.Pairwise('dpd_thermostat', rc, kind = "DPD", a = 0.0, gamma = gamma_dpd, kBT = kbt, power = s)
    dpd_wat = mir.Interactions.Pairwise('dpd_wat', rc, kind = "DPD", a = strong * aii, gamma = gamma_dpd, kBT = kbt, power = s)
    dpd = mir.Interactions.Pairwise('dpd', rc, kind = "DPD", a = aii, gamma = gamma_dpd, kBT = kbt, power = s)
    dpd_strong = mir.Interactions.Pairwise('dpd_strong', rc, kind = "DPD", a = strong * aii, gamma = gamma_dpd, kBT = kbt, power = s)
    lj = mir.Interactions.Pairwise('lj', rc, kind = "RepulsiveLJ", epsilon = 0.1, sigma = rc / (2**(1/6)), max_force = 10.0, aware_mode = 'Object')

    ######################################## INTEGRATOR ########################################
    #initialize integrator
    vv = mir.Integrators.VelocityVerlet('vv')

    #register integrator
    u.registerIntegrator(vv)

    #set integrator for various parts
    u.setIntegrator(vv, emb)
    u.setIntegrator(vv, water)
    u.setIntegrator(vv, gas)

    ######################################## INTERACTIONS ########################################
    #register interactions
    u.registerInteraction(int_emb)
    u.registerInteraction(dpd_thermostat)
    u.registerInteraction(dpd_wat)
    u.registerInteraction(dpd)
    u.registerInteraction(dpd_strong)
    u.registerInteraction(lj)

    #set interaction
    u.setInteraction(int_emb, emb, emb)
    u.setInteraction(dpd_wat, water, water)
    u.setInteraction(dpd, gas, gas)
    u.setInteraction(dpd, water, gas)
    u.setInteraction(dpd_strong, emb, water)
    u.setInteraction(dpd, emb, gas)
    u.setInteraction(lj, emb, emb)

    ######################################## REFLECTION BOUNDARIES ########################################
    #reflection boundaries of gas vesicle shells
    bouncer = mir.Bouncers.Mesh("membrane_bounce", "bounce_maxwell", kBT = kbt)
    u.registerBouncer(bouncer)
    u.setBouncer(bouncer, emb, water)
    u.setBouncer(bouncer, emb, gas)

    ######################################## RUN ########################################

    #u.registerPlugins(mir.Plugins.createStats('stats', every = nevery))
    #u.registerPlugins(mir.Plugins.createDumpXYZ('xyz_dump', emb, nevery, f"trj_eq/sim{args.simnum}"))
    #u.registerPlugins(mir.Plugins.createDumpObjectStats('objStats', emb, nevery, filename = 'stats/object' + args.simnum))
    u.run(numsteps, dt = dt)

    u.restart("restart/")
    #u.registerPlugins(mir.Plugins.createStats('stats', every = nevery))
    u.registerPlugins(mir.Plugins.createDumpXYZ('xyz_dump', emb, nevery, f"trj_eq/sim{args.simnum}"))
    #u.registerPlugins(mir.Plugins.createDumpObjectStats('objStats', emb, nevery, filename = 'stats/object' + args.simnum))
    u.run(numsteps, dt = dt)
        
    del u

def mirheo_pbuckling():
    """
    Compute the buckling pressure by running multiple simulations with increasing pressures
    """

    return(pbuckling)

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