import numpy as np

constants = {
    # Physical units
    'rho_water' : 997,     # water density, kg/m^3
    'kb'        : 1.3805e-23,     # Boltzmann constant, J/K
    'T25'       : 25,            # Reference temperature, °C

    # DPD parameters
    'nd'        : 3.0,            # Number density (DPD units, 1/L**3)
    'kBT_s'     : 1.0,        # Energy scale (DPD units, J)
    'm'         : 1.0,             # Mass of a DPD bead (DPD units, M)
    'rc'        : 1.0,            # Cutoff radius (DPD units, L)

    # Simulation parameters
    'L'         : 20.0,            # Box size (DPD units, L)
    'Fx'        : 0.01,           # Poiseuille force (DPD units, F)
}

# Compute the scaling units
size_water = 2.75e-10 # Size of a water molecule
Nm         = 8.0 # Number of water molecules in a DPD bead
ul         = size_water * Nm**(1/3) # Length scale (m)
#ul         = # 0.5e-6 #35e-9/1.0 # real/simu : 35nm = standard length of a gas vesicle 
um         = constants['rho_water']*(ul**3) / constants['nd']
ue         = constants['kb']*(constants['T25']+273.15) / constants['kBT_s']
ut         = np.sqrt(um*ul**2/ue)

units = {
    # Scaling units
    'L_UNIT'   : ul, # Length scale (Dimensionless)
    'M_UNIT'   : um, # Mass scale (Dimensionless)
    'E_UNIT'   : ue, # Energy scale (Dimensionless)
    'T_UNIT'   : ut, # Time scale (Dimensionless)
    'RHO_UNIT' : constants['rho_water'] / constants['nd'], # Density scale (Dimensionless)
    'ETA_UNIT' : um/(ul*ut), # Viscosity scale (Dimensionless)
}

obmd = {
    'bufferSize'  : 0.2,
    'bufferAlpha' : 0.7,
    'bufferTau'   : 10.0,
    'ptan'        : 0.0,
}

def convertToDPDUnits(file, units):
    """
    Convert the data in the file to DPD units
    """

    # Load the data
    data = np.loadtxt(file, skiprows=1)

    # Convert the data to DPD units
    data[:,0] = data[:,0] / units['RHO_UNIT']
    data[:,1] = data[:,1] / units['ETA_UNIT']

    # Write the data to a new file
    np.savetxt(file[:-4]+'_DPD.dat', data, header='Density Viscosity')

def convertToDPDUnitsDensitySpeed(file, units):
    """
    Convert the data in the file to DPD units
    """

    # Load the data
    data = np.loadtxt(file, skiprows=1)

    # Convert the data to DPD units
    data[:,0] = data[:,0] / units['RHO_UNIT']
    data[:,1] = data[:,1] / (units['L_UNIT']/units['T_UNIT'])

    # Write the data to a new file
    np.savetxt(file[:-4]+'_DPD.dat', data, header='Density Speed')