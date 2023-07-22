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

    # Getting box dimensions
    Lx, Ly, Lz = L

    # Loading dictionary of atomic masses

    atomic_masses = {}
    charges = {}
    electronegativities = {}
    ionization_energies = {}
    with open('/Users/cibran/Work/UPC/VASP/atomic_masses.dat', 'r') as atomic_masses_file:
        for line in atomic_masses_file:
            (key, mass, charge, electronegativity, ionization_energy) = line.split()
            atomic_masses[key] = mass
            charges[key] = charge
            electronegativities[key] = electronegativity
            ionization_energies[key] = ionization_energy

    # Counting number of particles
    total_particles = np.sum(concentration)

    # Getting particle types
    particle_types = []
    for i in range(len(composition)):
        particle_types += [i] * concentration[i]

    # Getting all nodes in the supercell
    nodes = []
    positions = []
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

        # Applying periodic boundary conditions (in reduced coordinates)
        while np.any(position_0 > 0.5):
            position_0[position_0 > 0.5] -= 1

        alpha_i = 1
        alpha_j = 1
        alpha_k = 1
        i = 0
        break_i = False
        while True:
            j = 0
            break_j = False
            while True:
                k = 0
                break_k = False
                while True:
                    # Moving to the corresponding image
                    position = position_0 + [i, j, k]

                    # Converting to cartesian distances
                    position_cartesian = np.dot(position, cell)

                    # If the cartesian coordinates belong to the imposed box, it is added to the list
                    if np.all(position_cartesian >= 0) and np.all(position_cartesian < [Lx, Ly, Lz]):
                        nodes.append(node)
                        positions.append(position_cartesian)
                        distance = 0
                    else:
                        distancex = np.min(np.abs(position_cartesian[0]), np.abs(position_cartesian[0] - Lx))
                        distancey = np.min(np.abs(position_cartesian[1]), np.abs(position_cartesian[1] - Ly))
                        distancez = np.min(np.abs(position_cartesian[2]), np.abs(position_cartesian[2] - Lz))
                        new_distance = distancex + distancey + distancez

                        # If new distance is smaller than before, k advances in alpha_k direction; else, we change direction or start again
                        if new_distance < distance:
                            distance = new_distance
                            k += alpha_k
                        else:
                            distance = -1

                            # If alpha_k is negative, k-search is finished; else, alpha_k is negative and it starts in zero
                            if alpha_k = 1:
                                alpha_k = -1
                                k = -1
                            else:
                                # Got to extreme for k
                                alpha_k = 1
                                break_k = True

                                # Got to extreme for j
                                if k == 0:
                                    if alpha_j = 1:
                                        alpha_j = -1
                                    else:
                                        break_j = True

                                # Got to extreme for i
                                if j == 0:
                                    if alpha_i = 1:
                                        alpha_i = -1
                                    else:
                                        break_i = True

                    # Updating k
                    if break_k: break
                # Updating j
                j += alpha_j
                if break_j: break
            # Updating i
            i += alpha_i
            if break_i: break

    # Keep track of the number of connections for each node
    n_connections = np.zeros(total_particles)

    # For each node, look for the three closest particles so that each node only has three connections
    for index_0 in range(total_particles):
        # Compute the distance of the current particle to all the others
        distances = np.linalg.norm(positions - positions[index_0])

        # Generate indexes, to easily keep track of the distance
        idxs = np.arange(total_particles)

        # Delete the distance to itself
        idxs      = np.delete(idxs,      index_0)
        distances = np.delete(distances, index_0)

        # Generate

        # Obtain the indexes of three closest ones wrt distances
        idx_min = np.argmin(distances)[:3]

        # Obtain corresponding distances and indexes wrt nodes
        idxs      = idxs[idx_min]
        distances = distances[idx_min]

        for




    nodes = torch.tensor(nodes, dtype=torch.float)
    edges = torch.tensor(edges, dtype=torch.long)
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
    edge_list = torch.tensor([])

    for data in dataset:
        target_list = torch.cat((target_list, data.y), 0)
        edge_list = torch.cat((edge_list, data.edge_attr), 0)

    scale = 1e0

    target_mean = torch.mean(target_list)
    target_std = torch.std(target_list)

    edge_mean = torch.mean(edge_list)
    edge_std = torch.std(edge_list)

    target_factor = target_std / scale
    edge_factor = edge_std / scale

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
    edge_attr = torch.randn(n_edges, 1)

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
    graph_t.x = torch.sqrt(alpha_t) * graph_t.x + torch.sqrt(1 - alpha_t) * epsilon.x
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

    # Backard pass
    graph_0.x = graph_0.x / torch.sqrt(alpha_t) - torch.sqrt((1 - alpha_t) / alpha_t) * epsilon.x
    graph_0.edge_attr = graph_0.edge_attr / torch.sqrt(alpha_t) - torch.sqrt(
        (1 - alpha_t) / alpha_t) * epsilon.edge_attr
    return graph_0


class nGCNN(torch.nn.Module):
    """Graph convolution neural network for the prediction of node embeddings.
    """

    def __init__(self, features_channels, pdropout):
        super(nGCNN, self).__init__()

        # Set random seed for reproducibility
        torch.manual_seed(12345)

        # Define graph convolution layers
        self.conv1 = GraphConv(features_channels, 512)
        self.conv2 = GraphConv(512, 512)

        # Define linear layers
        self.linconv = Linear(512, 16)
        self.lin = Linear(16, 1)

        self.pdropout = pdropout

    def forward(self, x, edge_index, edge_attr, batch):
        ## CONVOLUTION

        # Apply graph convolution with ReLU activation function
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        return x


class eGCNN(nn.Module):
    """Graph convolution neural network for the prediction of edge attributes.
    """

    def __init__(self, features_channels, pdropout):
        super(eGCNN, self).__init__()

        self.linear1 = Linear(features_channels, 32)
        self.linear2 = Linear(32, features_channels)

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