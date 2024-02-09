import numpy as np
import sys
import torch
import matplotlib.pyplot as plt

from scipy.interpolate                     import interp1d
from pymatgen.analysis.diffraction.xrd     import XRDCalculator
from pymatgen.analysis.diffraction.neutron import NDCalculator

# Checking if pytorch can run in GPU, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_diffraction_pattern(structure, diffraction='x-ray', theta_angles=np.arange(0, 90, 0.1)):
    """Computes the diffraction pattern of some crystal structure.
    It is implemented for x-ray of neutron diffraction patterns.

    Main changes by using neutron instead of X-ray are as follows:
    1. Atomic scattering length is a constant.
    2. Polarization correction term of Lorentz factor is unnecessary.

    Args:
        structure    (Pymatgen Structure object): reference crystal structure.
        diffraction  (str):                       x-ray or neutron diffraction.
        theta_angles (np.ndarray):                angles at which the diffraction pattern is computed.
    Returns:
        pattern (list): diffraction pattern for diffraction technique.
    """

    # Call calculator
    if diffraction == 'x-ray':
        calculator = XRDCalculator()
    elif diffraction == 'neutron':
        calculator = NDCalculator()
    else:
        sys.exit('Error: diffraction must be either "x-ray" or "neutron".')

    # Extract pattern
    # scaled=False in order to compare patterns across materials, if wanted
    pattern = calculator.get_pattern(structure, scaled=False)

    # Interpolate the theta angle at which the pattern has been calculated
    interpolation     = interp1d(pattern.x, pattern.y, fill_value='extrapolate')
    interpolated_data = interpolation(theta_angles)

    # Plot and compare interpolations and original pattern
    #plt.plot(pattern.x, pattern.y, 'ob')
    #plt.plot(theta_angles, interpolated_data, 'or')
    #plt.show()
    return list(interpolated_data)
