#!/usr/bin/env python
import unittest
import argparse
import numpy as np

import GM_library as GML

class TestIdentifyDiffusions(unittest.TestCase):
    """Class for testing the reading of simulation parameters.
    """
    
    # INCAR file reading
    
    def test_composition_concentration(self):
        """Checks that both composition and concentration for a XDATCAR is correctly extracted, and the positions subsequently disposed.
        """
        
        # Define some random keys
        keys = ['Li', 'La', 'La', 'La', 'Br', 'Li', 'Br', 'Br', 'Br', 'Cs', 'Li', 'Li']
        
        # Define some random positions
        direct_positions = np.array([[0.4815918 , 0.88894609, 0.78469354],
                                     [0.46399951, 0.34715816, 0.09911166],
                                     [0.13555168, 0.80503159, 0.80730379],
                                     [0.81718019, 0.87593676, 0.58689829],
                                     [0.31849491, 0.0521545 , 0.1805421 ],
                                     [0.39374104, 0.07275077, 0.54418534],
                                     [0.80410919, 0.57569525, 0.67511755],
                                     [0.19574369, 0.06456628, 0.55550622],
                                     [0.18161946, 0.37822116, 0.11685423],
                                     [0.24484669, 0.70017074, 0.0189568 ],
                                     [0.61721841, 0.32283448, 0.92101633],
                                     [0.9382648 , 0.2012692 , 0.05203284]])
        
        # Define composition and concentratio, and sort positions
        composition, concentration, positions_sorted = GML.composition_concentration_from_keys(keys, direct_positions)
        
        # Define correct results
        correct_composition      = ['Br', 'Cs', 'La', 'Li']
        correct_concentration    = [4, 1, 3, 4]
        correct_positions_sorted = np.array([[0.31849491, 0.0521545 , 0.1805421 ],
                                             [0.80410919, 0.57569525, 0.67511755],
                                             [0.19574369, 0.06456628, 0.55550622],
                                             [0.18161946, 0.37822116, 0.11685423],
                                             [0.24484669, 0.70017074, 0.0189568 ],
                                             [0.46399951, 0.34715816, 0.09911166],
                                             [0.13555168, 0.80503159, 0.80730379],
                                             [0.81718019, 0.87593676, 0.58689829],
                                             [0.4815918 , 0.88894609, 0.78469354],
                                             [0.39374104, 0.07275077, 0.54418534],
                                             [0.61721841, 0.32283448, 0.92101633],
                                             [0.9382648 , 0.2012692 , 0.05203284]])
        
        # Check the results
        self.assertEqual(composition,
                         correct_composition)
        self.assertEqual(concentration,
                         correct_concentration)
        self.assertEqual(positions_sorted.tolist(),
                         correct_positions_sorted.tolist())
