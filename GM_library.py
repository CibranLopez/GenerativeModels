import numpy               as np
import matplotlib.pyplot   as plt
import re                  as re
import torch               as torch
import torch.nn.functional as F
import sys                 as sys
import torch.nn            as nn
import yaml

from os                   import mkdir, path
from torch_geometric.data import Data
from torch.nn             import Linear
from torch_geometric.nn   import GCNConv, GraphConv
from torch_geometric.nn   import global_mean_pool

# Checking if pytorch can run in GPU, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_atoms_in_box(particle_types, composition, cell, atomic_masses, charges, electronegativities, ionization_energies, positions, L):
    """Create a list with all nodes and their positions inside the rectangular box.

    Args:
        particle_types      (list): type of particles (0, 1...).
        atomic_masses       (dict):
        charges             (dict):
        electronegativities (dict):
        ionization_energies (dict):
        positions:
        L                   (list): distances of the box in each direction.

    Returns:
        all_nodes     (list): features of each node in the box.
        all_positions (list): positions of the respective nodes.
    """

    # Getting box dimensions
    Lx, Ly, Lz = L

    #print('Starting loop')

    # Getting all nodes in the supercell
    all_nodes     = []
    all_positions = []
    for idx in range(len(particle_types)):
        # Get particle type (index of type wrt composition in POSCAR)
        particle_type = particle_types[idx]

        # Name of the current species
        species_name = composition[particle_type]

        # Loading the node (mass, charge, electronegativity anbd ionization energy)
        node = [float(atomic_masses[species_name]),
                float(charges[species_name]),
                float(electronegativities[species_name]),
                float(ionization_energies[species_name])
                ]

        # Get the initial position
        position_0 = positions[idx]

        #print(f'Particle {idx} of type {species_name}')

        # Get all images of particle_0 inside the intended box
        distance_i = None  # So first time it tries to get closer to the box
        distance_j = None
        distance_k = None
        i = 0
        alpha_i = 1
        break_i = False
        while True:
            j = 0
            alpha_j = 1
            break_j = False
            while True:
                k = 0
                alpha_k = 1
                break_k = False
                while True:
                    # Moving to the corresponding image
                    position = position_0 + [i, j, k]

                    # Converting to cartesian distances
                    position_cartesian = np.dot(position, cell)

                    #print()
                    #print(f'[i, j, k] = {i, j, k} at {position_cartesian}')

                    # If the cartesian coordinates belong to the imposed box, it is added to the list
                    if np.all(position_cartesian >= 0) and np.all(position_cartesian < [Lx, Ly, Lz]):
                        #print('Verified: into the box')
                        all_nodes.append(node)
                        all_positions.append(position_cartesian)
                        distance_k = 0
                        k += alpha_k
                    else:
                        distancex = np.min([np.abs(position_cartesian[0]), np.abs(position_cartesian[0] - Lx)])
                        distancey = np.min([np.abs(position_cartesian[1]), np.abs(position_cartesian[1] - Ly)])
                        distancez = np.min([np.abs(position_cartesian[2]), np.abs(position_cartesian[2] - Lz)])
                        new_distance = distancex + distancey + distancez

                        #print(f'Not verified: from {distance_k} to {new_distance}')

                        # If new distance is smaller than before or no initialized, k advances in alpha_k direction
                        # Else, we change direction or start again
                        if (distance_k is None) or (new_distance < distance_k):
                            #print('Continue')
                            distance_k = new_distance
                            k += alpha_k
                        else:
                            #print('Exit')
                            distance_k = None  # Initilizing it agains

                            # If alpha_k is negative, k-search is finished; else, alpha_k is negative and it starts in zero
                            if alpha_k == 1:
                                #print('Going backward')
                                alpha_k = -1
                                k = -1
                            else:
                                #print('Breaking k')
                                # Got to extreme for k
                                break_k = True  # Which puts k = 0 and alpha_k = 1

                                if (distance_j is None) or (new_distance < distance_j):
                                    #print('Continue')
                                    distance_j = new_distance
                                    j += alpha_j
                                else:
                                    #print('Exit')
                                    distance_j = None  # Initilizing it agains

                                    # If alpha_j is negative, j-search is finished; else, alpha_j is negative and it starts in zero
                                    if alpha_j == 1:
                                        #print('Going backward')
                                        alpha_j = -1
                                        j = -1
                                    else:
                                        #print('Breaking j')
                                        # Got to extreme for j
                                        break_j = True  # Which puts j = 0 and alpha_j = 1

                                        if (distance_i is None) or (new_distance < distance_i):
                                            #print('Continue')
                                            distance_i = new_distance
                                            i += alpha_i
                                        else:
                                            #print('Exit')
                                            distance_i = None  # Initilizing it agains

                                            # If alpha_i is negative, i-search is finished; else, alpha_i is negative and it starts in zero
                                            if alpha_i == 1:
                                                #print('Going backward')
                                                alpha_i = -1
                                                i = -1
                                            else:
                                                #print('Breaking i')
                                                # Got to extreme for i
                                                break_i = True  # Which puts i = 0 and alpha_i = 1
                    # Updating k
                    if break_k: break
                # Updating j
                j += alpha_j
                if break_j: break
            # Updating i
            i += alpha_i
            if break_i: break
    return all_nodes, all_positions


def get_edges_in_box(nodes, positions):
    """From a list of nodes and corresponding positions, get all edges and attributes for the graph.
    Every pair of particles are linked.

    Args:
        nodes (list): all nodes in the box.
        positions (list): corresponding positions of the particles.

    Returns:
        edges      (list): edges linking all pairs of nodes.
        attributes (list): weights of the corresponding edges (euclidean distance).
    """

    # Get total particles in the box
    total_particles = len(nodes)

    # For each node, look for the three closest particles so that each node only has three connections
    edges = []
    attributes = []
    for index_0 in range(total_particles):
        # Compute the distance of the current particle to all the others
        distances = np.linalg.norm(positions - positions[index_0], axis=1)

        # Generate indexes, to easily keep track of the distance
        idxs = np.arange(total_particles)

        # Delete the distance to itself
        idxs = np.delete(idxs, index_0)
        distances = np.delete(distances, index_0)

        # Add all edges
        edges.append([np.ones(len(idxs)) * index_0, idxs])
        attributes.append(distances)

    # Concatenating
    edges      = np.concatenate(edges, axis=1)  # Maintaining the order
    edges  # Transposing
    attributes = np.concatenate(attributes)  # Just distance for the previous pairs of links
    return edges, attributes

def graph_POSCAR_encoding(cell, composition, concentration, positions, L):
    """Generates a graph parameters from a POSCAR.
    Fills the space of a cubic box of dimension [0-Lx, 0-Ly, 0-Lz] considering all necessary images.
    It links every particle with the three closest ones within the box, disregarding images.

    Args:
        cell          (3x3 numpy array):       Lattice vectors of the reference POSCAR.
        composition   (3 numpy str array):     Names of the elements of the material.
        concentration (3 numpy int array):     Number of each of the previous elements.
        positions     (Nx3 numpy float array): Direct coordinates of each particle in the primitive cell.
        L             (3 numpy float array):   Maximum distance in each cartesian direction for describing the material.
    Returns:
        nodes      (torch tensor): Generated nodes with corresponding features.
        edges      (torch tensor): Generated connections between nodes.
        attributes (torch tensor): Corresponding weights of the generated connections.
    """

    # Loading dictionary of atomic masses

    atomic_masses       = {}
    charges             = {}
    electronegativities = {}
    ionization_energies = {}
    with open('/Users/cibran/Work/UPC/VASP/atomic_masses.dat', 'r') as atomic_masses_file:
        for line in atomic_masses_file:
            (key, mass, charge, electronegativity, ionization_energy) = line.split()
            atomic_masses[key]       = mass
            charges[key]             = charge
            electronegativities[key] = electronegativity
            ionization_energies[key] = ionization_energy

    # Counting number of particles
    POSCAR_particles = np.sum(concentration)

    # Getting particle types
    particle_types = []
    for i in range(len(composition)):
        particle_types += [i] * concentration[i]

    # Applying periodic boundary conditions (in reduced coordinates)
    # This is not strictly necessary (all images are being checked eitherway), but it can simplify things
    while np.any(positions > 0.5):
        positions[positions > 0.5] -= 1
    while np.any(positions < 0.5):
        positions[positions < 0.5] += 1

    # Load all nodes and respective positions in the box
    all_nodes, all_positions = get_atoms_in_box(particle_types,
                                                composition,
                                                cell,
                                                atomic_masses,
                                                charges,
                                                electronegativities,
                                                ionization_energies,
                                                positions,
                                                L)

    # Get edges and attributes for the corresponding nodes
    edges, attributes = get_edges_in_box(all_nodes, all_positions)

    # Convert to torch tensors and return
    nodes      = torch.tensor(all_nodes,  dtype=torch.float)
    edges      = torch.tensor(edges,      dtype=torch.long)
    attributes = torch.tensor(attributes, dtype=torch.float)
    return nodes, edges, attributes


def standardize_dataset(dataset, labels):
    """Stardizes a given dataset (both nodes features and edge attributes).
    Typically, a normal distribution is applied, although it be easily modified to apply other distributions.

    Currently: normal distribution.

    Args:
        dataset (list): List containing graph structures.
        labels  (list): List containing the label of each graph in the dataset.

    Returns:
        dataset    (list): Normalized dataset.
        parameters (list): Parameters needed to re-scale predicted properties from the dataset.
    """

    # Compute means and standard deviations

    target_list = torch.tensor([])
    edge_list   = torch.tensor([])

    for data in dataset:
        target_list = torch.cat((target_list, data.y), 0)
        edge_list   = torch.cat((edge_list, data.edge_attr), 0)

    scale = 1e0

    target_mean = torch.mean(target_list)
    target_std  = torch.std(target_list)

    edge_mean = torch.mean(edge_list)
    edge_std  = torch.std(edge_list)

    target_factor = target_std / scale
    edge_factor   = edge_std / scale

    # Update normalized values into the database

    for data in dataset:
        data.y = (data.y - target_mean) / target_factor
        data.edge_attr = (data.edge_attr - edge_mean) / edge_factor

    # Same for the node features

    feat_mean = torch.tensor([])
    feat_std = torch.tensor([])

    for feat_index in range(dataset[0].num_node_features):
        feat_list = torch.tensor([])

        for data in dataset:
            feat_list = torch.cat((feat_list, data.x[:, feat_index]), 0)

        feat_mean = torch.cat((feat_mean, torch.tensor([torch.mean(feat_list)])), 0)
        feat_std = torch.cat((feat_std, torch.tensor([torch.std(feat_list)])), 0)

        for data in dataset:
            data.x[:, feat_index] = (data.x[:, feat_index] - feat_mean[feat_index]) * scale / feat_std[feat_index]

    parameters = [target_mean, feat_mean, edge_mean, target_std, edge_std, feat_std, scale]
    return dataset, parameters


def get_alpha_t(t, T, s=1e-2):
    """Defines constant alpha at time-step t, given a parameter s < 0.5 (else alpha increases).

    Args:
        t (int):   time step (of diffusion or denoising) in which alpha is required.
        T (int):   total number of steps.
        s (float): parameter which controls the decay of alpha with t.

    Returns:
        alpha (float): parameter which controls the velocity of diffusion or denoising.
    """

    return torch.tensor((1 - 2 * s) * (1 - (t / T) ** 2) + s)


def get_random_graph(n_nodes, n_features, in_edge_index=None, n_edges=None):
    """Generates a random graph with specified number of nodes and features, and attributes. It is assumed
    that all parameters are normally distributed N(0, 1).

    Args:
        n_nodes       (int):   Number of nodes.
        n_features    (int):   Number of features for each node.
        in_edge_index (array): Positions of high-symmetry points in k-space (if None, they are randomized).
        n_edges       (int):   Number of edges, if edge_index is randomized (if None, it is randomized).

    Returns:
        graph (torch_geometric.data.data.Data): Graph structure with random node features and edge attributes.
    """

    if in_edge_index is None:  # Randomize edge indexes
        if n_edges is None:  # Randomize number of edges
            n_edges = torch.randint(low=50, high=101, size=(1,)).item()
        edge_index = torch.randn(2, n_edges)
    else:
        # Clone edge indexes
        edge_index = torch.clone(in_edge_index)

    # Get number of edges
    n_edges = torch.Tensor.size(edge_index)[1]

    # Generate random node features
    x = torch.randn(n_nodes, n_features)

    # Generate random edge attributes
    edge_attr = torch.randn(n_edges)

    # Define graph with generated inputs
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return graph


def diffusion_step(graph_0, t, n_diffusing_steps):
    """Performs a forward step of a diffusive, Markov chain.

    Args:
        graph_0 (torch_geometric.data.data.Data): Graph which is to be diffused (step t-1).

    Returns:
        graph_t (torch_geometric.data.data.Data): Diffused graph (step t).
    """

    # Clone graph that we are diffusing (not extrictly necessary)
    graph_t = graph_0.clone()

    # Number of nodes and features in the graph
    n_nodes, n_features = torch.Tensor.size(graph_t.x)

    # Generate gaussian (normal) noise
    epsilon = get_random_graph(n_nodes, n_features, graph_t.edge_index)

    # Compute alpha_t
    alpha_t = get_alpha_t(t, n_diffusing_steps)

    # Forward pass
    graph_t.x         = torch.sqrt(alpha_t) * graph_t.x         + torch.sqrt(1 - alpha_t) * epsilon.x
    graph_t.edge_attr = torch.sqrt(alpha_t) * graph_t.edge_attr + torch.sqrt(1 - alpha_t) * epsilon.edge_attr
    return graph_t


def diffuse(graph_0, n_diffusing_steps):
    """Performs consecutive steps of diffusion in a reference graph.

    Args:
        graph_0           (torch_geometric.data.data.Data): Reference graph to be diffused (step t-1).
        n_diffusing_steps (int):                            Number of diffusive steps.

    Returns:
        graph_t (torch_geometric.data.data.Data): Graph with random node features and edge attributes (step t).
    """

    graph_t = graph_0.clone()
    for t in range(n_diffusing_steps):
        graph_t = diffusion_step(graph_t, t, n_diffusing_steps)
    return graph_t


def denoising_step(graph_t, epsilon, t, n_denoising_steps):
    """Performs a forward step of a denoising chain.

    Args:
        graph_t (torch_geometric.data.data.Data): Graph which is to be denoised (step t).
        epsilon (torch_geometric.data.data.Data): Predicted noise to substract.

    Returns:
        graph_0 (torch_geometric.data.data.Data): Denoised graph (step t-1).
    """

    # Clone graph that we are denoising (not extrictly necessary)
    graph_0 = graph_t.clone()

    # Compute alpha_t
    alpha_t = get_alpha_t(t, n_denoising_steps)
    print(graph_0, epsilon)
    # Backard pass
    graph_0.x         = graph_0.x         / torch.sqrt(alpha_t) - torch.sqrt((1 - alpha_t) / alpha_t) * epsilon.x
    graph_0.edge_attr = graph_0.edge_attr / torch.sqrt(alpha_t) - torch.sqrt((1 - alpha_t) / alpha_t) * epsilon.edge_attr
    return graph_0


class nGCNN(torch.nn.Module):
    """Graph convolution neural network for the prediction of node embeddings.
    """

    def __init__(self, features_channels, pdropout):
        super(nGCNN, self).__init__()

        # Set random seed for reproducibility
        torch.manual_seed(12345)

        # Define graph convolution layers
        self.conv1 = GraphConv(features_channels, 64)  # Introducing node features
        self.conv2 = GraphConv(64, 64)
        self.conv3 = GraphConv(64, features_channels)  # Predicting node features

        self.pdropout = pdropout

    def forward(self, x, edge_index, edge_attr):
        # Apply graph convolution with ReLU activation function
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)
        return x


class eGCNN(nn.Module):
    """Graph convolution neural network for the prediction of edge attributes.
    """

    def __init__(self, features_channels, pdropout):
        super(eGCNN, self).__init__()

        self.linear1 = Linear(features_channels, 32)  # Introducing node features
        self.linear2 = Linear(32, 1)  # Predicting one single weight

        self.pdropout = pdropout

    def forward(self, x_i, x_j):
        # Dot product between node distances (?)
        x = x_i * x_j

        # Linear convolutions
        x = self.linear1(x)
        x = x.relu()

        # Dropout layer (only for training)
        x = F.dropout(x, p=self.pdropout, training=self.training)

        # Last linear convolution
        x = self.linear2(x)
        x = x.relu()
        return x