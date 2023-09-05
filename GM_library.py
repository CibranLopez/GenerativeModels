import numpy               as np
import matplotlib.pyplot   as plt
import torch.nn.functional as F
import torch.nn            as nn
import networkx            as nx
import torch
import sys

from os                            import mkdir, path
from torch_geometric.data          import Data
from torch.nn                      import Linear
from torch_geometric.nn            import GCNConv, GraphConv
from torch_geometric.nn            import global_mean_pool
from torch_geometric.utils.convert import to_networkx

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
    all_species   = []
    all_skipped   = []
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
        
        
        i = 0
        alpha_i = 1
        while True:
            minimum_distance_j = None
            success_j = False
            j = 0
            alpha_j = 1
            while True:
                minimum_distance_k = None
                reference_distance = np.NaN  # So it outputs False when first compared with another distance
                success_k = False
                k = 0
                alpha_k = 1
                while True:
                    # Move to the corresponding image and convert to cartesian distances
                    position_cartesian = np.dot(position_0 + [i, j, k], cell)
                    
                    new_distance = get_distance_to_box(position_cartesian, L)
                    if new_distance == 0:
                        #print('Verified: into the box')
                        all_nodes.append(node)
                        all_positions.append(position_cartesian)
                        all_species.append(species_name)
                        
                        # Check if successful
                        success_k = True
                    
                    # Change direction or update i,j if the box is far
                    elif new_distance > reference_distance:
                        # Explore other direction or cancel
                        if alpha_k == 1:
                            k = 0
                            alpha_k = -1
                        else: break
                    
                    reference_distance = new_distance
                    k += alpha_k
                    
                    if (minimum_distance_k is None) of (minimum_distance_k > reference_distance):
                        minimum_distance_k = reference_distance
                
                # If k worked fine, j is fine as well thus continue; else, explore other direction or cancel
                if success_k:
                    success_j = True
                else:
                    if alpha_j == 1:
                        j = 0
                        alpha_j = -1
                    else: break
                
                # Update j
                j += alpha_j
            
            # If j did not work fine, explore other direction or cancel
            if not success_j:
                if alpha_i == 1:
                    i = 0
                    alpha_i = -1
                else: break
            
            # Update i
            i += alpha_i
    return all_nodes, all_positions, all_species, all_skipped


def get_distance_to_box(position_cartesian, L):
    """Computes the euclidean distance between a given point and a box of shape [Lx, Ly, Lz].
    
    Args:
        position_cartesian (ndarray) Cartesian coordinates of the point.
        L                  (list):   Length of the box.
    
    Returns:
        distance (float): Euclidean distance between point and box.
    """
        
    distance = 0
    for index in range(3):
        if position_cartesian[index] < 0:
            distance += np.power(position_cartesian[index], 2)
        elif position_cartesian[index] > L[index]:
            distance += np.power(position_cartesian[index] - L[index], 2)
    distance = np.sqrt(distance)
    
    return distance


def get_edges_in_box(nodes, positions):
    """From a list of nodes and corresponding positions, get all edges and attributes for the graph.
    Every pair of particles are linked.

    Args:
        nodes     (list): All nodes in the box.
        positions (list): Corresponding positions of the particles.

    Returns:
        edges      (list): Edges linking all pairs of nodes.
        attributes (list): Weights of the corresponding edges (euclidean distance).
    """

    # Get total particles in the box
    total_particles = len(nodes)
    
    # Generate indexes, to easily keep track of the distance
    idxs = np.arange(total_particles)
    
    # For each node, look for the three closest particles so that each node only has three connections
    edges = []
    attributes = []
    for index_0 in range(total_particles - 1):
        # Compute the distance of the current particle to all the others
        distances = np.linalg.norm(positions - positions[index_0], axis=1)

        # Delete distances above the current index (avoiding repeated distances)
        temp_idxs = idxs[index_0+1:]
        distances = distances[index_0+1:]

        # Add all edges
        edges.append([np.ones(len(temp_idxs)) * index_0, temp_idxs])
        attributes.append(distances)

    # Concatenating
    edges      = np.concatenate(edges, axis=1)  # Maintaining the order
    attributes = np.concatenate(attributes)  # Just distance for the previous pairs of links
    return edges, attributes


def graph_POSCAR_encoding(cell, composition, concentration, positions, L):
    """Generates a graph parameters from a POSCAR.
    Fills the space of a cubic box of dimension [0-Lx, 0-Ly, 0-Lz] considering all necessary images.
    It links every particle with the three closest ones within the box, disregarding images.

    Args:
        cell          (3x3 float ndarray): Lattice vectors of the reference POSCAR.
        composition   (3   str   ndarray): Names of the elements of the material.
        concentration (3   int   ndarray): Number of each of the previous elements.
        positions     (Nx3 float ndarray): Direct coordinates of each particle in the primitive cell.
        L             (3   float ndarray): Maximum distance in each cartesian direction for describing the material.
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
    #while np.any(positions > 1):
    #    positions[positions > 1] -= 1
    #while np.any(positions < 0):
    #    positions[positions < 0] += 1

    # Load all nodes and respective positions in the box
    all_nodes, all_positions, all_species, all_skipped = get_atoms_in_box(particle_types,
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
    return nodes, edges, attributes, all_nodes, all_positions, all_species, all_skipped


def standardize_dataset(dataset, labels):
    """Standardizes a given dataset (both nodes features and edge attributes).
    Typically, a normal distribution is applied, although it be easily modified to apply other distributions.

    Currently: normal distribution.

    Args:
        dataset (list): List containing graph structures.
        labels  (list): List containing the label of each graph in the dataset.

    Returns:
        dataset_std (list): Normalized dataset.
        parameters  (list): Parameters needed to re-scale predicted properties from the dataset.
    """
    
    # Clone the dataset (using a list comprehension)
    dataset_std = [graph.clone() for graph in dataset]
    
    # Compute means and standard deviations

    target_list = torch.tensor([])
    edge_list   = torch.tensor([])

    for data in dataset_std:
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

    for data in dataset_std:
        data.y = (data.y - target_mean) / target_factor
        data.edge_attr = (data.edge_attr - edge_mean) / edge_factor

    # Same for the node features

    feat_mean = torch.tensor([])
    feat_std = torch.tensor([])

    for feat_index in range(dataset_std[0].num_node_features):
        feat_list = torch.tensor([])

        for data in dataset_std:
            feat_list = torch.cat((feat_list, data.x[:, feat_index]), 0)

        feat_mean = torch.cat((feat_mean, torch.tensor([torch.mean(feat_list)])), 0)
        feat_std = torch.cat((feat_std, torch.tensor([torch.std(feat_list)])), 0)

        for data in dataset_std:
            data.x[:, feat_index] = (data.x[:, feat_index] - feat_mean[feat_index]) * scale / feat_std[feat_index]

    parameters = [target_mean, feat_mean, edge_mean, target_std, edge_std, feat_std, scale]
    return dataset_std, parameters


def revert_standardize_dataset(dataset, parameters):
    """De-standardizes a given dataset (both nodes features and edge attributes).
    Typically, a normal distribution is applied, although it be easily modified to apply other distributions.

    Currently: normal distribution.

    Args:
        dataset    (list): List containing graph structures.
        parameters (list): Parameters needed to re-scale predicted properties from the dataset.

    Returns:
        dataset_rstd (list): De-normalized dataset.
    """
    
    # Clone the dataset (using a list comprehension)
    dataset_rstd = [graph.clone() for graph in dataset]
    
    # Assigning parameters accordingly
    _, feat_mean, edge_mean, _, edge_std, feat_std, scale = parameters
    
    edge_factor = edge_std / scale

    # Update normalized values into the database
    for data in dataset_rstd:
        data.edge_attr = data.edge_attr * edge_factor + edge_mean

    # Same for the node features
    for feat_index in range(dataset_rstd[0].num_node_features):
        for data in dataset_rstd:
            data.x[:, feat_index] = data.x[:, feat_index] * feat_std[feat_index] / scale + feat_mean[feat_index]

    return dataset_rstd


def get_alpha_t(t, T, s):
    """Defines constant alpha at time-step t, given a parameter s < 0.5 (else alpha increases).
    
    \alpha (t) = (1 - 2 s) \left( 1 - \left( \frac{t}{T} \right)^2 \right) + s

    Args:
        t (int):   time step (of diffusion or denoising) in which alpha is required.
        T (int):   total number of steps.
        s (float): parameter which controls the decay of alpha with t.

    Returns:
        alpha (float): parameter which controls the velocity of diffusion or denoising.
    """

    return torch.tensor((1 - 2 * s) * (1 - (t / T) ** 2) + 2 * s)


def get_random_graph(n_nodes, n_features, in_edge_index=None):
    """Generates a random graph with specified number of nodes and features, and attributes. It is assumed
    that all parameters are normally distributed N(0, 1).

    Args:
        n_nodes       (int):   Number of nodes.
        n_features    (int):   Number of features for each node.
        in_edge_index (array): Positions of high-symmetry points in k-space (if None, they are randomized).

    Returns:
        graph (torch_geometric.data.Data): Graph structure with random node features and edge attributes.
    """
    
    if in_edge_index is None:  # Randomize edge indexes
        # Randomize number of edges
        edge_index = []
        
        # Generate indexes, to easily keep track of the distance
        idxs = np.arange(n_nodes)
        for index_0 in range(n_nodes - 1):
            # Delete distances above the current index (avoiding repeated distances)
            temp_idxs = idxs[index_0+1:]

            # Add all edges
            edge_index.append([np.ones(len(temp_idxs)) * index_0, temp_idxs])
        
        # Concatenating
        edge_index = np.concatenate(edge_index, axis=1)  # Maintaining the order
        
        # Convert to torch tensor
        edge_index = torch.tensor(edge_index, dtype=torch.long)
    else:
        # Clone edge indexes
        edge_index = torch.clone(in_edge_index)

    # Generate random node features
    x = torch.randn(n_nodes, n_features)

    # Get number of edges
    n_edges = torch.Tensor.size(edge_index)[1]
    
    # Generate random edge attributes
    edge_attr = torch.randn(n_edges)

    # Define graph with generated inputs
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return graph


def diffusion_step(graph_0, t, n_diffusing_steps, s):
    """Performs a forward step of a diffusive, Markov chain.
    
    G (t) = \sqrt{\alpha (t)} G (t-1) + \sqrt{1 - \alpha (t)} N (t)
    
    with G a graph and N noise.

    Args:
        graph_0 (torch_geometric.data.Data): Graph which is to be diffused (step t-1).

    Returns:
        graph_t (torch_geometric.data.Data): Diffused graph (step t).
    """

    # Clone graph that we are diffusing (not extrictly necessary)
    graph_t = graph_0.clone()

    # Number of nodes and features in the graph
    n_nodes, n_features = torch.Tensor.size(graph_t.x)

    # Generate gaussian (normal) noise
    epsilon = get_random_graph(n_nodes, n_features, graph_t.edge_index)

    # Compute alpha_t
    alpha_t = get_alpha_t(t, n_diffusing_steps, s)

    # Forward pass
    graph_t.x         = torch.sqrt(alpha_t) * graph_t.x         + torch.sqrt(1 - alpha_t) * epsilon.x
    graph_t.edge_attr = torch.sqrt(alpha_t) * graph_t.edge_attr + torch.sqrt(1 - alpha_t) * epsilon.edge_attr
    return graph_t


def diffuse(graph_0, n_diffusing_steps, s=1e-2, plot_steps=False):
    """Performs consecutive steps of diffusion in a reference graph.

    Args:
        graph_0           (torch_geometric.data.Data): Reference graph to be diffused (step t-1).
        n_diffusing_steps (int):                       Number of diffusive steps.
        s                 (float):                     Parameter which controls the decay of alpha with t.
        plot_steps        (bool):                      Whether to plot or not each intermediate step.

    Returns:
        graph_t (torch_geometric.data.Data): Graph with random node features and edge attributes (step t).
    """
    
    graph_t = graph_0.clone()
    
    # Save all intermediate graphs as well (sanity check)
    all_graphs = [graph_t]
    
    # Define t_steps starting from 1 to n_denoising_steps+1
    t_steps = np.arange(1, n_diffusing_steps+1)
    for t in t_steps:
        # Check if intermediate steps are plotted; then, plot the NetworkX graph
        if plot_steps:
            # Convert PyTorch graph to NetworkX graph
            networkx_graph = to_networkx(graph_t)
            pos            = nx.spring_layout(networkx_graph)
            nx.draw(networkx_graph, pos, with_labels=True, node_size=graph_t.x, font_size=10)
            plt.show()
        
        graph_t = diffusion_step(graph_t, t, n_diffusing_steps, s)
        all_graphs.append(graph_t)
    
    # Check if intermediate steps are plotted; then, plot the NetworkX graph
    if plot_steps:
        # Convert PyTorch graph to NetworkX graph
        networkx_graph = to_networkx(graph_t)
        pos            = nx.spring_layout(networkx_graph)
        nx.draw(networkx_graph, pos, with_labels=True, node_size=graph_t.x, font_size=10)
        plt.show()
    return graph_t, all_graphs


def denoise(graph_t, n_denoising_steps, node_model, edge_model, s=1e-2, plot_steps=False):
    """Performs consecutive steps of diffusion in a reference graph.

    Args:
        graph_t           (torch_geometric.data.Data): Reference graph to be diffused (step t-1).
        n_diffusing_steps (int):                       Number of diffusive steps.
        s                 (float):                     Parameter which controls the decay of alpha with t.
        plot_steps        (bool):                      Whether to plot or not each intermediate step.

    Returns:
        graph_0 (torch_geometric.data.Data): Graph with random node features and edge attributes (step t).
    """
    
    graph_0 = graph_t.clone()
    
    # Save all intermediate graphs as well (sanity check)
    all_graphs = [graph_0]
    
    # Define t_steps as inverse of the diffuse process
    t_steps = np.arange(1, n_denoising_steps+1)[::-1]
    for t in t_steps:
        # Perform a single forward pass for predicting node features
        out_x = node_model(graph_0.x,
                           graph_0.edge_index,
                           graph_0.edge_attr)
        
        # Define x_i and x_j as features of every corresponding pair of nodes (same order than attributes)
        x_i = graph_0.x[graph_0.edge_index[0]]
        x_j = graph_0.x[graph_0.edge_index[1]]
        
        # Perform a single forward pass for predicting edge attributes
        out_attr = edge_model(x_i, x_j)

        # Construct noise graph
        noise_graph = Data(x=out_x, edge_index=graph_0.edge_index, edge_attr=out_attr.ravel())
        
        # Check if intermediate steps are plotted; then, plot the NetworkX graph
        if plot_steps:
            # Convert PyTorch graph to NetworkX graph
            networkx_graph = to_networkx(graph_0)
            pos            = nx.spring_layout(networkx_graph)
            nx.draw(networkx_graph, pos, with_labels=True, node_size=graph_0.x, font_size=10)
            plt.show()
        
        # Denoise the graph with the predicted noise
        graph_0 = denoising_step(graph_0, noise_graph, t, n_denoising_steps, s=s)
        all_graphs.append(graph_0)
        
    # Check if intermediate steps are plotted; then, plot the NetworkX graph
    if plot_steps:
        # Convert PyTorch graph to NetworkX graph
        networkx_graph = to_networkx(graph_0)
        pos            = nx.spring_layout(networkx_graph)
        nx.draw(networkx_graph, pos, with_labels=True, node_size=graph_0.x, font_size=10)
        plt.show()
    return graph_0, all_graphs


def denoising_step(graph_t, epsilon, t, n_denoising_steps, s):
    """Performs a forward step of a denoising chain.

    Args:
        graph_t (torch_geometric.data.Data): Graph which is to be denoised (step t).
        epsilon (torch_geometric.data.Data): Predicted noise to substract.

    Returns:
        graph_0 (torch_geometric.data.Data): Denoised graph (step t-1).
    """

    # Clone graph that we are denoising (not extrictly necessary)
    graph_0 = graph_t.clone()

    # Compute alpha_t
    alpha_t = get_alpha_t(t, n_denoising_steps, s)

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


def find_closest_key(dictionary, target_array):
    """
    Find the key in the dictionary that corresponds to the array closest to the target array.

    Parameters:
        dictionary   (dict):          A dictionary where keys are associated with arrays.
        target_array (numpy.ndarray): The target array for comparison.

    Returns:
        str: The key corresponding to the closest array in the dictionary.
    """
    
    closest_key = None
    closest_distance = float('inf')

    # Iterate through the dictionary
    for key, array in dictionary.items():
        # Calculate the Euclidean distance between the current array and target array
        distance = np.linalg.norm(array - target_array)
        
        # Update the closest key and distance if the current distance is smaller
        if distance < closest_distance:
            closest_distance = distance
            closest_key = key

    return closest_key


def discretize_graph(graph):
    """
    Convert the graph's continuous node embeddings to the closest valid embeddings based on the periodic table.

    Args:
        graph (torch_geometric.data.Data): The initial graph structure with continuous node embeddings.

    Returns:
        new_graph (torch_geometric.data.Data): The modified graph with closest valid node embeddings based on the periodic table.
    """

    # Clone the input graph to preserve the original structure
    new_graph = graph.clone()

    # Detach embeddings for the graph nodes
    data_embeddings = new_graph.x.detach()

    # Load the dictionary of available embeddings for atoms
    available_embeddings = {}
    with open('../VASP/atomic_masses.dat', 'r') as atomic_masses_file:
        for line in atomic_masses_file:
            (key, mass, charge, electronegativity, ionization_energy) = line.split()

            # Check if some information is missing
            if ((mass              == 'None') or
                (charge            == 'None') or
                (electronegativity == 'None') or
                (ionization_energy == 'None')):
                continue

            # Add valid atom embeddings to the library
            available_embeddings[key] = np.array([mass, charge, electronegativity, ionization_energy], dtype=float)

    # Iterate through each graph node to update embeddings
    for i in range(new_graph.num_nodes):
        # Load the original embedding for the current node
        old_embedding = new_graph.x[i].detach().cpu().numpy()

        # Find the closest key (atom) from available_embeddings for the old_embedding
        key = find_closest_key(available_embeddings, old_embedding)

        # Get the valid atom embeddings corresponding to the closest key
        new_embeddings = available_embeddings[key]

        # Update the embeddings for the current node in the new graph
        new_graph.x[i] = torch.tensor(new_embeddings)
    
    # Calculate the loss for node features and edge attributes
    node_loss, edge_loss = get_graph_losses(graph, new_graph)
    
    # Accumulate the total training loss
    loss = node_loss + edge_loss
    train_loss = loss.item()
    return new_graph


def composition_concentration_from_keys(keys, positions):
    """
    Calculate composition and concentration from a list of keys. It sorts the elements, so they are enumerated only once. Attending to that, the positions are sorted as well.

    Args:
        keys      (list):       A list of keys representing some data.
        positions (np.ndarray): Position of each key (element).

    Returns:
        composition   (list): A list of composition values.
        concentration (list): A list of concentration values.
    """
    
    # Sort the keys
    indexes          = np.argsort(keys)
    keys_sorted      = np.sort(keys)
    positions_sorted = positions[indexes]
    
    # Get unique values and their frequencies from the list of keys
    unique_values = np.unique(keys_sorted, return_counts=True)
    
    # Extract composition and concentration from the unique values
    composition   = unique_values[0].tolist()
    concentration = unique_values[1].tolist()
    
    return composition, concentration, positions_sorted


def POSCAR_graph_encoding(graph, L, POSCAR_name=None, POSCAR_directory='./'):
    """
    Encode a graph into a POSCAR (VASP input) file format.

    Args:
        graph            (torch_geometric.data.Data): The input graph structure with continuous node embeddings.
        L                (list):                      Lattice parameters assumed to be orthogonal, in the format [a, b, c].
        POSCAR_name      (str, optional):             Name for the POSCAR file. Defaults to None.
        POSCAR_directory (str, optional):             Directory for the POSCAR file to be saved at. Defaults to current folder.

    Returns:
        file: A file object representing the generated POSCAR file.
    """
    
    # Check validity of the graph (if it defines a real material)
    check_graph_validity(graph)
    
    # Get name for the first line of the POSCAR
    POSCAR_name = POSCAR_name or 'POSCAR from GenerativeModels'

    # Clone the input graph to preserve the original structure
    new_graph = graph.clone()

    # Load and detach embeddings for the graph nodes
    data_embeddings = new_graph.x.detach().cpu().numpy()

    # Loading dictionary of available embeddings for atoms
    available_embeddings = {}
    with open('../VASP/atomic_masses.dat', 'r') as atomic_masses_file:
        for line in atomic_masses_file:
            key, mass, charge, electronegativity, ionization_energy = line.split()

            # Check if all information is present
            if all(val != 'None' for val in (mass, charge, electronegativity, ionization_energy)):
                available_embeddings[key] = np.array([mass, charge, electronegativity, ionization_energy], dtype=float)

    # Get most similar atoms for each graph node and create a list of keys
    keys = [find_closest_key(available_embeddings, emb) for emb in data_embeddings]

    # Define the lattice vectors from L vector
    lattice_vectors = [
        [L[0], 0.0,  0.0],
        [0.0,  L[1], 0.0],
        [0.0,  0.0,  L[2]],
    ]
    
    # Check consistency of the graph
    ...

    # Get the position of each atom in direct coordinates
    direct_positions = graph_to_direct_positions(graph, L)

    # Get elements' composition, concentration, and positions
    POSCAR_composition, POSCAR_concentration, POSCAR_positions = composition_concentration_from_keys(keys, direct_positions)

    # Write file
    with open(f'{POSCAR_directory}/POSCAR', 'w') as POSCAR_file:
        # Delete previous data in the file
        POSCAR_file.truncate()
        
        # Write POSCAR's name
        POSCAR_file.write(f'{POSCAR_name}\n')

        # Write scaling factor (assumed to be 1.0)
        POSCAR_file.write('1.0\n')

        # Write lattice parameters (assumed to be orthogonal)
        np.savetxt(POSCAR_file, lattice_vectors, delimiter=' ')

        # Write composition (each different species, previously sorted)
        np.savetxt(POSCAR_file, [POSCAR_composition], fmt='%s', delimiter=' ')

        # Write concentration (number of each of the previous elements)
        np.savetxt(POSCAR_file, [POSCAR_concentration], fmt='%d', delimiter=' ')

        # Write position in direct form
        POSCAR_file.write('Direct\n')
        np.savetxt(POSCAR_file, POSCAR_positions, delimiter=' ')

    return POSCAR_file


def get_graph_losses(graph1, graph2):
    """
    Calculate loss values for node features and edge attributes between two graphs.

    Args:
        graph1 (torch_geometric.data.Data): The first input graph.
        graph2 (torch_geometric.data.Data): The second input graph.

    Returns:
        node_loss (torch.Tensor): Loss value for node features between the two graphs.
        edge_loss (torch.Tensor): Loss value for edge attributes between the two graphs.
    """

    # Initialize loss criterions for nodes and edges
    node_criterion = nn.MSELoss()
    edge_criterion = nn.MSELoss()

    # Calculate the loss for node features by comparing the node attribute tensors
    node_loss = node_criterion(graph1.x,
                               graph2.x)

    # Calculate the loss for edge attributes by comparing the edge attribute tensors
    edge_loss = edge_criterion(graph1.edge_attr,
                               graph2.edge_attr)

    return node_loss, edge_loss


def allocate_atom_n(d_01, x2, y2, d_0n, d_1n, d_2n):
    """
    Calculate the coordinates of atom 'n' based on geometric constraints.

    Args:
        d_01 (float): Distance between atoms '0' and '1'.
        x2   (float): x-coordinate of atom '2'.
        y2   (float): y-coordinate of atom '2'.
        d_0n (float): Distance between atoms '0' and 'n'.
        d_1n (float): Distance between atoms '1' and 'n'.
        d_2n (float): Distance between atoms '2' and 'n'.

    Returns:
        list: A list containing the x, y, and z coordinates of atom 'n'.
    """
    
    # Calculate x-coordinate of atom 'n'
    xn = (d_01**2 + d_0n**2 - d_1n**2) / (2 * d_01)
    
    # Calculate y-coordinate of atom 'n'
    yn = (d_1n**2 - d_2n**2 - d_01**2 + 2 * xn * d_01 + x2**2 - 2 * xn * x2 + y2**2) / (2 * y2)
    
    # Calculate z-coordinate of atom 'n'
    zn_square = d_0n**2 - xn**2 - yn**2
    zn = np.sqrt(zn_square) if zn_square > 0 else 0
    
    return [xn, yn, zn]


def get_distance_attribute(index0, index1, edge_indexes, edge_attributes):
    """
    Get the distance attribute between two nodes with given indices.

    Args:
        index0          (int): Index of the first node.
        index1          (int): Index of the second node.
        edge_indexes    (np.ndarray): Array containing indices of connected nodes for each edge.
        edge_attributes (np.ndarray): Array containing attributes corresponding to each edge.

    Returns:
        float: The distance attribute between the two nodes.
        
    Raises:
        SystemExit: If the provided node indices do not correspond to a linked pair.
    """
    
    # Create a mask to find matching edges
    mask_direct  = (edge_indexes[0] == index0) & (edge_indexes[1] == index1)
    mask_reverse = (edge_indexes[0] == index1) & (edge_indexes[1] == index0)
    
    # Check if any edge satisfies the conditions
    matching_edge_indices = np.where(mask_direct | mask_reverse)[0]
    
    if len(matching_edge_indices) == 0:
        sys.exit(f'Error: the pair [{index0}, {index1}] is not linked')
    
    # Get the distance attribute from the first matching edge
    distance_attribute = edge_attributes[matching_edge_indices[0]]
    
    return distance_attribute


def graph_to_direct_positions(graph, L):
    """
    Calculate the positions of atoms in a molecular graph based on given distances, in direct coordinates.
    The graph is assumed to be self-consistent (all lenghts to be correct).

    Args:
        graph (torch_geometric.data.Data): The input graph containing edge indexes and attributes.

    Returns:
        list: A list of atom positions in the format [x, y, z]. Dimensionless, with [x, y, z] in (0, 1).
    """
    
    # Extract indexes and attributes from the graph
    edge_indexes    = graph.edge_index.detach().cpu().numpy()
    edge_attributes = graph.edge_attr.detach().cpu().numpy()

    # Define the number of atoms in the graph
    total_particles = len(graph.x)
    
    # Find three indexes for using as reference of the graph
    index_0, index_1, index_2 = find_three_indexes(edge_indexes, edge_attributes, total_particles)
    
    # Get necessary distances
    d_01 = get_distance_attribute(index_0, index_1, edge_indexes, edge_attributes)
    d_02 = get_distance_attribute(index_0, index_2, edge_indexes, edge_attributes)
    d_12 = get_distance_attribute(index_1, index_2, edge_indexes, edge_attributes)

    # Reference the first three atoms
    x2 = (d_01**2 + d_02**2 - d_12**2) / (2 * d_01)
    y2 = np.sqrt(d_02**2 - x2**2)
    
    # Iterate over the remaining atoms (positions regarding the graph are preserved)
    cartesian_positions = []
    for n in range(total_particles):
        if   n == index_0: rn = [0,    0,  0]
        elif n == index_1: rn = [d_01, 0,  0]
        elif n == index_2: rn = [x2,   y2, 0]
        else:
            # Get distances for atom 'n'
            d_0n = get_distance_attribute(index_0, n, edge_indexes, edge_attributes)
            d_1n = get_distance_attribute(index_1, n, edge_indexes, edge_attributes)
            d_2n = get_distance_attribute(index_2, n, edge_indexes, edge_attributes)
            
            # Allocate atom 'n' using optimized function
            rn = allocate_atom_n(d_01, x2, y2, d_0n, d_1n, d_2n)
        
        # Append the new position to the positions list
        cartesian_positions.append(rn)
    
    # Pass to numpy ndarray
    direct_positions = np.array(cartesian_positions)
    
    # Pass from cartesian to direct coordinates
    direct_positions[:, 0] /= L[0]
    direct_positions[:, 1] /= L[1]
    direct_positions[:, 2] /= L[2]
    
    # Translate between 0 and 1
    #while np.any(direct_positions >= 1):
    #    direct_positions[direct_positions >= 1] -= 1
    #while np.any(direct_positions < 0):
    #    direct_positions[direct_positions < 0] += 1
    
    return direct_positions


def check_graph_validity(graph):
    """Check that the current graph describes a realistic material (positive interatomic distances, etc.).

    Args:
        graph (torch_geometric.data.Data): The input graph containing edge indexes and attributes.
        
    Raises:
        SystemExit: If the provided graph has negative or null interatomic distances.
    """
    
    if torch.any(graph.edge_attr <= 0):
        sys.exit('Invalid graph, atoms overlapping')


def get_target_loss(obtained_target, seeked_target):
    """
    Calculate the target loss based on obtained and seeked targets. It checks if seeked_target is a specific value or rather a limit

    Args:
        obtained_target (float):        The obtained target value.
        seeked_target   (float or str): The desired target value or a string indicating the type of target.

    Returns:
        float: The calculated target loss.
    """
    
    if np.isscalar(seeked_target):
        # Target loss defined as the absolute difference between obtained and seeked targets
        return np.abs(obtained_target - seeked_target)
    
    elif seeked_target == 'positive':
        # Return a large negative value to indicate the target is positive
        return seeked_target * (-1)
    
    elif seeked_target == 'negative':
        # Return a positive value to indicate the target is negative
        return seeked_target
    
    # If none of the above conditions are met, raise an error
    sys.exit('Error: the seeked target is not valid')


def triangle_area(a, b, c):
    """Compute the area of a triangle.
    
    Args:
        a (float): Lenght of one edge.
        b (float): Lenght of one edge.
        c (float): Lenght of one edge.
    
    Returns:
        area (float): Area of the triangle.
    """
    
    # Calculate the semi-perimeter
    s = (a + b + c) / 2
    
    # Calculate the area using Heron's formula
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    
    return area


def find_three_indexes(edge_indexes, edge_attributes, total_particles, threshold=10):
    """Finds three nodes which define a triangle of appropiate area.
    
    Args:
        edge_indexes    (array): List of edge indexes.
        edge_attributes (array): Corresponding list of edge attributes.
        total_particles (int):   Total number of particles in the graph.
    
    Returns:
        indexes (int): Indexes of the three nodes used as base.
    """
    
    indexes = (0, 0, 0)  # Initialize with dummy values
    for i in np.arange(total_particles):
        for j in np.arange(i+1, total_particles):
            for k in np.arange(j+1, total_particles):
                d_ij = get_distance_attribute(i, j, edge_indexes, edge_attributes)
                d_jk = get_distance_attribute(j, k, edge_indexes, edge_attributes)
                d_ik = get_distance_attribute(i, k, edge_indexes, edge_attributes)
                
                d_temp = triangle_area(d_ij, d_jk, d_ik)
                if d_temp > threshold:
                    return (i, j, k)
