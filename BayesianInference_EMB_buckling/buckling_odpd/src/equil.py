#!/usr/bin/env python3

import mirheo as mir
import numpy as np
import trimesh
import yaml
import argparse
import os

######################################################
# set-up simulation type: equilibration or restart

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--equil', action = 'store_true', default = None)
group.add_argument('--restart', action = 'store_true', default = None)
parser.add_argument('--simnum', dest = 'simnum', default = '00001')

args = parser.parse_args()

dir_name = 'restart'
if(args.restart):
	if os.path.isdir(dir_name):
		if not os.listdir(dir_name):
			print("Directory restart is empty. Changing simulation mode from --restart to --equil.")
			args.restart = False
			args.equil = True
	else:
		args.restart = False
		args.equil = True
		print("Given directory doesn't exist. Changing simulation mode from --restart to --equil.")

######################################################
# set-up parameters

filename_default = 'parameter/parameters-default' + args.simnum + '.yaml'
with open(filename_default, 'rb') as f:
    parameters_default = yaml.load(f, Loader = yaml.CLoader)
    
filename = 'parameter/parameters' + args.simnum + '.yaml'
with open(filename, 'rb') as f:
    parameters = yaml.load(f, Loader = yaml.CLoader)    

filename_prms = 'parameter/parameters.prms' + args.simnum + '.yaml'
with open(filename_prms, 'rb') as f:
    prms_emb = yaml.load(f, Loader = yaml.CLoader)

objFile = parameters_default["objFile"]
numsteps = (int(parameters_default["numsteps"]) if args.restart else int(parameters_default["numsteps_eq"]))
numsteps_eq = int(parameters_default["numsteps_eq"])
dt = (parameters_default["dt"] if args.restart else parameters_default["dt_eq"])
dt_eq = parameters_default["dt_eq"]
Lx = parameters_default["Lx"]
Ly = parameters_default["Ly"]
Lz = parameters_default["Lz"]
nevery = (parameters["nevery"] if args.restart else parameters["nevery_eq"])
rhow = parameters_default["rhow"]
rhog = parameters_default["rhog"]
alpha = parameters_default["alpha"]
aii = parameters_default["aii"] * parameters["kbt"]
strong = parameters_default["strong"]
gamma_dpd = parameters_default["gamma_dpd"]
gamma_fsi = parameters["gamma_fsi"]
rc = parameters_default["rc"]
s = parameters_default["s"] 
kbt = parameters["kbt"]
obmd_flag = parameters_default["obmd_flag"]
mvert = parameters["mvert"]
mw = parameters["mw"]
mg = parameters["mg"]
buck = parameters_default["buck"]

timestart = (numsteps_eq * dt_eq if args.restart else 0.0)
timeend = (numsteps_eq * dt_eq + numsteps * dt if args.restart else numsteps_eq * dt_eq)

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

tdump = nevery * dt

#interactions
int_emb = mir.Interactions.MembraneForces("int_emb", "Lim", "KantorStressFree", **prms_emb, stress_free = True) 
dpd_thermostat = mir.Interactions.Pairwise('dpd_thermostat', rc, kind = "DPD", a = 0.0, gamma = gamma_dpd, kBT = kbt, power = s)
dpd_wat = mir.Interactions.Pairwise('dpd_wat', rc, kind = "DPD", a = buck * aii, gamma = gamma_dpd, kBT = kbt, power = s)
dpd = mir.Interactions.Pairwise('dpd', rc, kind = "DPD", a = aii, gamma = gamma_dpd, kBT = kbt, power = s)
dpd_fsi = mir.Interactions.Pairwise('dpd_strong', rc, kind = "DPD", a = 0.0, gamma = gamma_fsi, kBT = kbt, power = s)
lj = mir.Interactions.Pairwise('lj', rc, kind = "RepulsiveLJ", epsilon = 0.1, sigma = rc / (2**(1/6)), max_force = 10.0, aware_mode = 'Object')
odpd = mir.Interactions.Pairwise('odpd', rc, kind = "ODPD", a = aii, gamma = gamma_dpd, kBT = kbt, power = s, timestart = timestart, timeend = timeend, amp = buck * aii, mode = 2, stress=True, stress_period = tdump) # 1 = 'hysteresis', 0 - 'forward', 2 - 'forward + equil'

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
u.registerInteraction(odpd)
u.registerInteraction(dpd_fsi)
u.registerInteraction(lj)

#set interaction
u.setInteraction(int_emb, emb, emb)
u.setInteraction(odpd, water, water)
u.setInteraction(dpd, gas, gas)
u.setInteraction(dpd, water, gas)
u.setInteraction(dpd_wat, emb, water)
u.setInteraction(dpd_wat, emb, gas)
u.setInteraction(lj, emb, emb)

######################################## REFLECTION BOUNDARIES ########################################
#reflection boundaries of gas vesicle shells
bouncer = mir.Bouncers.Mesh("membrane_bounce", "bounce_maxwell", kBT = kbt)
u.registerBouncer(bouncer)
u.setBouncer(bouncer, emb, water)
u.setBouncer(bouncer, emb, gas)

######################################## RUN ########################################

def predicate_all_domain(r):
	return 1.0

h = (1.0, 1.0, 1.0)

if args.equil:
    print('equilibration')
    u.registerPlugins(mir.Plugins.createStats('stats', every = nevery))
    u.registerPlugins(mir.Plugins.createDumpXYZ('xyz_dump', emb, nevery, f"trj_eq/sim{args.simnum}"))
    u.registerPlugins(mir.Plugins.createDumpXYZ('xyz_dump_gas', gas, nevery, f"trj_eq/sim{args.simnum}"))
    u.registerPlugins(mir.Plugins.createVirialPressurePlugin('virial', water, predicate_all_domain, h, nevery, 'pressure/p' + args.simnum))
    u.registerPlugins(mir.Plugins.createDumpObjectStats('objStats', emb, nevery, filename = 'stats/object' + args.simnum))
    u.run(numsteps, dt = dt)
    del u

if args.restart:
    print('productionn')
    u.restart("restart/")
    u.registerPlugins(mir.Plugins.createStats('stats', every = nevery))
    u.registerPlugins(mir.Plugins.createDumpXYZ('xyz_dump', emb, nevery, f"trj_eq/sim{args.simnum}"))
    u.registerPlugins(mir.Plugins.createVirialPressurePlugin('virial', water, predicate_all_domain, h, nevery, 'pressure/p' + args.simnum))
    u.registerPlugins(mir.Plugins.createDumpObjectStats('objStats', emb, nevery, filename = 'stats/object' + args.simnum))
    u.run(numsteps, dt = dt)
    del u
