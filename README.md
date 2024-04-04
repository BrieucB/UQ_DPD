# Inference of DPD parameters ($a$, $\gamma$ and $s$) for water using Korali and Mirheo

## Run the code on a Vrana15 node
1. `ssh` to a gpu-equiped Vrana15 node and clone the code from Github in your local folder: `$ git clone -b master git@github.com:BrieucB/UQ_DPD.git`
2. The code relies on two softwares: Korali and Mirheo. Both are compiled in a Singularity container located on o378. Before running an inference problem, first start a local instance of this container:
```
$ singularity instance start \
--nv --writable-tmpfs \
--no-home --bind .:/RUN /net/o378/temp/jaka/SingularMirheo/MKSingularity_MPL.sif SingMK
```
3. The code was built to take as input a reduced number of parameters, contained in a parameter file `metaparam.dat` (for more information, read next section). For the sake of simplicity, we are going to run a tMCMC optimization, for which a standard set of parameters can be written:
``` 
$ cat <<EOF > UQ_DPD/metaparam.dat
#L Fx rho_s kBT_s tmax pop_size
16.0 0.5 3.0 0.01 1000 1000
EOF
```
4. Finally, run the inference inside the Singularity in the background:
```
$ nohup singularity exec instance://SingMK \
/bin/bash -c "cd /RUN/UQ_DPD && mpirun -np 3 python3 runTMCMC.py" > /dev/null 2>&1 < /dev/null &
``` 
5. When the inference is completed, the details can be found in the subfolder `UQ_DPD/_korali_result_tmcmc/`. A plot `watch.png` containing the most important results (among which the marginal PDF) can be obtained by running the following command inside the root folder `UQ_DPD`:
```
singularity exec instance://SingMK /bin/bash -c "python3 -m korali.plot --dir _korali_result_tmcmc/ --output watch.png"
```
For more information, visit the [official documentation of Korali.](https://korali.readthedocs.io/en/v3.0.1/using/tools/plotter.html)


## Basic operation
Setup:
- Computational model: viscosity measurements using Mirheo with a double Poiseuille flow
- Data: viscosity of water at 25°C
- Inference: Korali implementation of CMAES or TMCMC solver for Bayesian Inference

The `metaparam.dat` file contains the following parameters:

- `L`: The size of the simulation box
- `Fx`: The external force applied to obtain a double Poiseuille flow
- `rho_s`: The density of particles in simulation units
- `kBT_s`: The energy scale of the simulation
- `tmax`: The maximum time for the simulation
- `pop_size`: The population size for the solver