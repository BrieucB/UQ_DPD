# Inference of DPD parameters for water using Korali and Mirheo

Setup:
- Computational model: viscosity measurements in DPD units using Mirheo with a double Poiseuille flow
- Data: viscosity of water at 25°C
- Inference: Korali implementation of CMAES or TMCMC solver for Bayesian Inference

The `metaparam.dat` file contains the following parameters:

- `L`: The size of the simulation box
- `Fx`: The external force applied to obtain a double Poiseuille flow
- `rho_s`: The density of particles in simulation units
- `kBT_s`: The energy scale of the simulation
- `tmax`: The maximum time for the simulation
- `pop_size`: The population size for the solver