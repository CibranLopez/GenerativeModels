import numpy               as np
import matplotlib.pyplot   as plt
import re                  as re
import torch               as torch
import torch.nn.functional as F
import sys                 as sys
import yaml

from os                 import mkdir, path
from torch.nn           import Linear
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.nn import global_mean_pool

# Checking if pytorch can run in GPU, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def graph_POSCAR_encoding(cell, composition, concentration, positions):
    """Encodes the primitive cell from a POSCAR into a pytorch graph structure.
    
    Args:
        cell ():
        composition
        concentration
        positions
    
    Returns:
        nodes (nn.): vector of features defining each atom.
        edges (): indexes of edges corresponding to next attributes (all atoms are connected withing the primitive cell).
        attributes (): euclidean distance between each pair of nodes.
    """

    # Loading dictionary of atomic masses

    atomic_masses = {}
    charges = {}
    electronegativities = {}
    ionization_energies = {}
    with open('../VASP/atomic_masses.dat', 'r') as atomic_masses_file:
        for line in atomic_masses_file:
            (key, mass, charge, electronegativity, ionization_energy) = line.split()
            atomic_masses[key] = mass
            charges[key] = charge
            electronegativities[key] = electronegativity
            ionization_energies[key] = ionization_energy

    # Counting number of particles

    total_particles = np.sum(concentration)

    # Generating graph structure, getting particle types

    particle_types = []
    for i in range(len(composition)):
        particle_types += [i] * concentration[i]

    # Adding nodes and edges

    index_0 = 0
    nodes = []
    edges = []
    attributes = []
    for particle_type in particle_types:
        # Name of the current species

        species_name = composition[particle_type]

        # Adding the nodes (mass and charge)

        nodes.append([float(atomic_masses[species_name]),
                      float(charges[species_name]),
                      float(electronegativities[species_name]),
                      float(ionization_energies[species_name]),
                      float(temperature)]
                      )

        # Adding the edges as the distance between the current particle and all others in the primitive cell
        # Images of itself are removed, so the loop starts at index_0+1 instead of index_0

        for index_i in np.arange(index_0+1, total_particles):
            # Computing the distance among particles
            distance_i = np.abs(positions[index_i] - positions[index_0])

            # Applying periodic boundary conditions (in reduced coordinates)
            distance_i[distance_i > 0.5] -= 1

            # Converting to cartesian distances
            distance_i = np.dot(distance_i, cell)

            # Computing norm and number of nearest neighbors
            distance_i = np.linalg.norm(distance_i)
            
            # Edges indexes
            edges.append([index_0, index_i])

            # Edges attributes
            attributes.append([distance_i])
            
        # Particle index
        index_0 += 1

    # Convert list structures into pytorch tensors
    nodes = torch.tensor(nodes, dtype=torch.float)
    edges = torch.tensor(edges, dtype=torch.long)
    attributes = torch.tensor(attributes, dtype=torch.float)
    return nodes, edges, attributes
