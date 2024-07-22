import numpy as np

units = {
    # Scaling units
    'L_UNIT'   : ul, # Length scale (Dimensionless)
    'M_UNIT'   : um, # Mass scale (Dimensionless)
    'E_UNIT'   : ue, # Energy scale (Dimensionless)
    'T_UNIT'   : ut, # Time scale (Dimensionless)
    'RHO_UNIT' : constants['rho_water'] / constants['nd'], # Density scale (Dimensionless)
    'ETA_UNIT' : um/(ul*ut), # Viscosity scale (Dimensionless)
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