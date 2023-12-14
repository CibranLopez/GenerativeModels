import numpy               as np
import matplotlib.pyplot   as plt
import torch.nn.functional as F
import torch.nn            as nn
import networkx            as nx
import torch
import sys
import itertools

from pymatgen.core.structure       import Structure
from scipy.spatial                 import Voronoi
from torch.nn                      import Linear
from torch_geometric.nn            import GCNConv, GraphConv

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
        positions           (list): direc coordinates of particles.
        L                   (list): size of the box in each direction (x, y, z).

    Returns:
        all_nodes     (list): features of each node in the box.
        all_positions (list): positions of the respective nodes.
    """

    # Getting all nodes in the supercell
    all_nodes     = []
    all_positions = []
    all_species   = []
    for idx in range(len(particle_types)):
        # Get particle type (index of type wrt composition in POSCAR)
        particle_type = particle_types[idx]

        # Name of the current species
        species_name = composition[particle_type]

        # Loading the node (mass, charge, electronegativity and ionization energy)
        node = [float(atomic_masses[species_name]),
                float(charges[species_name]),
                float(electronegativities[species_name]),
                float(ionization_energies[species_name])
                ]

        # Get the initial position
        position_0 = positions[idx]

        reference_distance_i = np.NaN  # So it outputs False when first compared with another distance
        i = 0
        alpha_i = 1
        while True:
            minimum_distance_i   = np.NaN
            reference_distance_j = np.NaN
            j = 0
            alpha_j = 1
            while True:
                minimum_distance_j   = np.NaN
                reference_distance_k = np.NaN
                k = 0
                alpha_k = 1
                while True:
                    # Move to the corresponding image and convert to cartesian distances
                    position_cartesian = np.dot(position_0 + [i, j, k], cell)
                    
                    new_distance = get_distance_to_box(position_cartesian, L)
                    if new_distance == 0:
                        # Append this particle as one inside the desired region
                        # Afterwards, we use this data to construct edge connections
                        all_nodes.append(node)
                        all_positions.append(position_cartesian)
                        all_species.append(species_name)
                    
                    # Change direction or update i,j if the box is far
                    elif new_distance > reference_distance_k:
                        # Explore other direction or cancel
                        if alpha_k == 1:
                            k = 0
                            alpha_k = -1
                        else:
                            break
                    
                    reference_distance_k = new_distance
                    k += alpha_k
                    
                    if not minimum_distance_j <= reference_distance_k:
                        minimum_distance_j = reference_distance_k
                
                # If k worked fine, j is fine as well thus continue; else, explore other direction or cancel
                if minimum_distance_j > reference_distance_j:
                    if alpha_j == 1:
                        j = 0
                        alpha_j = -1
                    else:
                        break
                
                # Update j
                j += alpha_j
                reference_distance_j = minimum_distance_j
                
                if not minimum_distance_i <= reference_distance_j:
                    minimum_distance_i = reference_distance_j
            
            # If j did not work fine, explore other direction or cancel
            if minimum_distance_i > reference_distance_i:
                if alpha_i == 1:
                    i = 0
                    alpha_i = -1
                else:
                    break
            
            # Update i
            i += alpha_i
            reference_distance_i = minimum_distance_i
    return all_nodes, all_positions, all_species


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


def get_atoms_in_unitcell(particle_types, composition, cell, atomic_masses, charges, electronegativities, ionization_energies, positions):
    """Create a list with all nodes and their positions belonging to the unit cell.

    Args:
        particle_types      (list): type of particles (0, 1...).
        atomic_masses       (dict):
        charges             (dict):
        electronegativities (dict):
        ionization_energies (dict):
        positions           (list): direc coordinates of particles.

    Returns:
        all_nodes     (list): features of each node in the box.
        all_positions (list): positions of the respective nodes.
    """

    # Getting all nodes in the supercell
    all_nodes     = []
    all_positions = []
    all_species   = []
    for idx in range(len(particle_types)):
        # Get particle type (index of type wrt composition in POSCAR)
        particle_type = particle_types[idx]

        # Name of the current species
        species_name = composition[particle_type]

        # Loading the node (mass, charge, electronegativity and ionization energy)
        node = [float(atomic_masses[species_name]),
                float(charges[species_name]),
                float(electronegativities[species_name]),
                float(ionization_energies[species_name])
                ]

        # Get the initial position
        position_0 = positions[idx]
        
        # Verify that belongs to the unit cell
        while np.any(position_0 >  1): position_0[np.where(position_0 > 1)]  -= 1
        while np.any(position_0 < -1): position_0[np.where(position_0 < -1)] += 1
        
        # Convert to cartesian coordinates
        position_cartesian_0 = np.dot(position_0, cell)

        # Append this particle, which belong to the unit cell
        # Afterwards, we use this data to construct edge connections
        all_nodes.append(node)
        all_positions.append(position_cartesian_0)
        all_species.append(species_name)
    return all_nodes, all_positions, all_species


def get_voronoi_tessellation(atomic_masses, charges, electronegativities, ionization_energies, structure):
    """
    Get the Voronoi nodes of a structure.
    Templated from the TopographyAnalyzer class, added to pymatgen.analysis.defects.utils by Yiming Chen, but now deleted.
    Modified to map down to primitive, do Voronoi analysis, then map back to original supercell; much more efficient.

    Args:
        structure (pymatgen Structure object): Structure from which the graph is to be generated
    """

    # Map all sites to the unit cell; 0 ≤ xyz < 1
    structure = Structure.from_sites(structure, to_unit_cell=True)

    # Get Voronoi nodes in primitive structure and then map back to the
    # supercell
    prim_structure = structure.get_primitive_structure()

    # Get all atom coords in a supercell of the structure because
    # Voronoi polyhedra can extend beyond the standard unit cell
    coords = []
    cell_range = list(range(-1, 2))
    for shift in itertools.product(cell_range, cell_range, cell_range):
        for site in prim_structure.sites:
            shifted = site.frac_coords + shift
            coords.append(prim_structure.lattice.get_cartesian_coords(shifted))

    # Voronoi tessellation
    voro = Voronoi(coords)
    
    tol = 1e-3
    edges      = []
    attributes = []
    for atoms in voro.ridge_points:  # Atoms are indexes referred to coords
        # Dictionary for storing information about each atom
        atoms_info = {}
        
        # Check if any of those atoms belong to the unitcell
        for atom_idx in range(2):
            atom = atoms[atom_idx]
            
            # Direct ccordinates from supercell referenced to the primitive cell
            frac_coords = prim_structure.lattice.get_fractional_coords(coords[atom])

            is_atom_inside = True
            frac_coords_uc = frac_coords
            if not np.all([-tol <= coord < 1 + tol for coord in frac_coords]):
                # atom_x is not inside
                is_atom_inside = False

                # Apply periodic bounday conditions
                while np.any(frac_coords_uc >  1): frac_coords_uc[np.where(frac_coords_uc > 1)]  -= 1
                while np.any(frac_coords_uc < -1): frac_coords_uc[np.where(frac_coords_uc < -1)] += 1
            
            # Obtain mapping to index in unit cell
            uc_idx = np.argmin(np.linalg.norm(structure.frac_coords - frac_coords_uc, axis=1))
            
            # Generate dictionary storing relevant information of the atom
            atom_info = {
                atom_idx: {
                    'atom':           atom,
                    'is_atom_inside': is_atom_inside,
                    'cart_coords':    coords[atom],
                    'uc_idx':         uc_idx
                }
            }
            # Update with information of current atom
            atoms_info.update(atom_info)
        
        # Check if any of those belong to the unitcell
        if atoms_info[0]['is_atom_inside'] or atoms_info[1]['is_atom_inside']:
            uc_idx_x = atoms_info[0]['uc_idx']
            uc_idx_y = atoms_info[1]['uc_idx']
            
            cart_coords_x = atoms_info[0]['cart_coords']
            cart_coords_y = atoms_info[1]['cart_coords']
            
            edges.append([uc_idx_x,
                          uc_idx_y])
            
            attributes.append(np.linalg.norm(cart_coords_x - cart_coords_y))

    # Generate nodes from all atoms in structure
    nodes = []
    for idx in range(structure.num_sites):
        # Get species type
        species_name = str(structure[idx].species)[:-1]

        # Get node info
        # Loading the node (mass, charge, electronegativity and ionization energy)
        node = [float(atomic_masses[species_name]),
                float(charges[species_name]),
                float(electronegativities[species_name]),
                float(ionization_energies[species_name])
                ]
        nodes.append(node)
    return nodes, edges, attributes


def get_all_linked_edges_and_attributes(nodes, positions):
    """From a list of nodes and corresponding positions, get all edges and attributes for the graph.
    Every pair of particles is linked.

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


def graph_POSCAR_encoding(structure, encoding_type='voronoi'):
    """Generates a graph parameters from a POSCAR.
    There are two implementations:
        1.- Voronoi tessellation.
        2.- Fills the space of a cubic box of dimension [0-Lx, 0-Ly, 0-Lz] considering all necessary images. It links every particle with the rest for the given set of nodes and edges.

    Args:
        structure (pymatgen Structure object): Structure from which the graph is to be generated.
        encoding_type (str): Framework used for encoding the structure.
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
    with open('../VASP/atomic_masses.dat', 'r') as atomic_masses_file:
        for line in atomic_masses_file:
            (key, mass, charge, electronegativity, ionization_energy) = line.split()
            atomic_masses[key]       = mass
            charges[key]             = charge
            electronegativities[key] = electronegativity
            ionization_energies[key] = ionization_energy

    if encoding_type == 'voronoi':
        # Get edges and attributes for the corresponding tessellation
        nodes, edges, attributes = get_voronoi_tessellation(atomic_masses,
                                                            charges,
                                                            electronegativities,
                                                            ionization_energies,
                                                            structure)
        
        # Convert to torch tensors and return
        nodes      = torch.tensor(nodes,      dtype=torch.float)
        edges      = torch.tensor(edges,      dtype=torch.long).T
        attributes = torch.tensor(attributes, dtype=torch.float)
        
    elif encoding_type == 'box':
        """
        # Get particle types as a list with each species occupying a new positions, maintaining order
        particle_types = []
        for i in range(len(composition)):
            particle_types += [i] * concentration[i]
        
        # Load all nodes and respective positions in the box
        nodes, positions, species = get_atoms_in_unitcell(particle_types,
                                                          composition,
                                                          cell,
                                                          atomic_masses,
                                                          charges,
                                                          electronegativities,
                                                          ionization_energies,
                                                          positions)

        # Get edges and attributes for the corresponding nodes (all linked to each other)
        edges, attributes = get_all_linked_edges_and_attributes(all_nodes, all_positions)
        
        # Convert to torch tensors and return
        nodes      = torch.tensor(nodes,      dtype=torch.float)
        edges      = torch.tensor(edges,      dtype=torch.long)
        attributes = torch.tensor(attributes, dtype=torch.float)
        """
    return nodes, edges, attributes


def find_closest_key(dictionary, target_array):
    """Find the key in the dictionary that corresponds to the array closest to the target array.

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
    """Convert the graph's continuous node embeddings to the closest valid embeddings based on the periodic table.

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
    """Calculate composition and concentration from a list of keys. It sorts the elements, so they are enumerated only once. Attending to that, the positions are sorted as well.

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


def POSCAR_graph_encoding(graph, lattice_vectors, file_name='POSCAR', POSCAR_name=None, POSCAR_directory='./'):
    """Encode a graph into a POSCAR (VASP input) file format.

    Args:
        graph            (torch_geometric.data.Data): The input graph structure with continuous node embeddings.
        lattice_vectors  (numpy arrray):              Lattice vectors.
        file_name        (str, optional):             Name for the POSCAR file. Defaults to POSCAR.
        POSCAR_name      (str, optional):             Title for the POSCAR file. Defaults to None.
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

    # Get the position of each atom in direct coordinates
    direct_positions = graph_to_cartesian_positions(graph)

    # Get elements' composition, concentration, and positions
    POSCAR_composition, POSCAR_concentration, POSCAR_positions = composition_concentration_from_keys(keys, direct_positions)

    # Write file
    with open(f'{POSCAR_directory}/{file_name}', 'w') as POSCAR_file:
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

        # Write position in cartesian form
        POSCAR_file.write('Cartesian\n')
        np.savetxt(POSCAR_file, POSCAR_positions, delimiter=' ')

    return POSCAR_file


def allocate_atom_n(d_01, x2, y2, d_0n, d_1n, d_2n):
    """Calculate the coordinates of atom 'n' based on geometric constraints.

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
    if zn_square > -1e-4:  # Accounting for numerical errors
        zn = np.sqrt(zn_square)
        return [xn, yn, zn]
    return None


def get_distance_attribute(index0, index1, edge_indexes, edge_attributes):
    """Get the distance attribute between two nodes with given indices.

    Args:
        index0          (int):        Index of the first node.
        index1          (int):        Index of the second node.
        edge_indexes    (np.ndarray): Array containing indices of connected nodes for each edge.
        edge_attributes (np.ndarray): Array containing attributes corresponding to each edge.

    Returns:
        float: The distance attribute between the two nodes.
        False if index0, index1 not linked.
    """
    
    # Create a mask to find matching edges
    mask_direct  = (edge_indexes[0] == index0) & (edge_indexes[1] == index1)
    mask_reverse = (edge_indexes[0] == index1) & (edge_indexes[1] == index0)
    
    # Check if any edge satisfies the conditions
    matching_edge_indices = np.where(mask_direct | mask_reverse)[0]
    
    if len(matching_edge_indices) == 0:
        return None  # The pair is not linked
    
    # Get the distance attribute from the first matching edge
    distance_attribute = edge_attributes[matching_edge_indices[0]]
    
    return distance_attribute


def find_initial_basis(total_particles, edge_indexes, edge_attributes):
    for idx_0 in range(total_particles):
        for idx_1 in np.arange(idx_0+1, total_particles):
            for idx_2 in np.arange(idx_1+1, total_particles):
                condition_01 = (get_distance_attribute(idx_0, idx_1, edge_indexes, edge_attributes) is not None)
                condition_12 = (get_distance_attribute(idx_1, idx_2, edge_indexes, edge_attributes) is not None)
                condition_02 = (get_distance_attribute(idx_0, idx_2, edge_indexes, edge_attributes) is not None)
                if condition_01 and condition_12 and condition_02:
                    return idx_0, idx_1, idx_2


def find_valid_reference(n_connected, idx_connected, edge_indexes, edge_attributes):
    for i in range(n_connected):
        for j in np.arange(i+1, n_connected):
            for k in np.arange(j+1, n_connected):
                idx_0 = idx_connected[i]
                idx_1 = idx_connected[j]
                idx_2 = idx_connected[k]

                x2, y2, _ = cartesian_positions[idx_2]
                
                # Get necessary distances
                d_01 = get_distance_attribute(idx_0, idx_1, edge_indexes, edge_attributes)
                d_0n = get_distance_attribute(idx_0, idx,   edge_indexes, edge_attributes)
                d_1n = get_distance_attribute(idx_1, idx,   edge_indexes, edge_attributes)
                d_2n = get_distance_attribute(idx_2, idx,   edge_indexes, edge_attributes)
                
                temp_position = allocate_atom_n(d_01, x2, y2, d_0n, d_1n, d_2n)
                
                if temp_position is not None:
                    return temp_position


def graph_to_cartesian_positions(graph):
    """Calculate the positions of atoms in a molecular graph based on given distances, in cartesian coordinates.
    The graph is assumed to be self-consistent (all lenghts to be correct), and with a Voronoi tessellation.

    Args:
        graph (torch_geometric.data.Data): The input graph containing edge indexes and attributes.

    Returns:
        dict: A dictionary of atom positions in the format [x, y, z] for each atomic index, in angstroms.
    """
    
    # Extract indexes and attributes from the graph
    edge_indexes    = graph.edge_index.detach().cpu().numpy()
    edge_attributes = graph.edge_attr.detach().cpu().numpy()

    # Define the number of atoms in the graph
    total_particles = len(graph.x)
    
    # Select three initial particles which are interconnected
    idx_0, idx_1, idx_2 = find_initial_basis(total_particles, edge_indexes, edge_attributes)
    
    # Get necessary distances
    d_01 = get_distance_attribute(idx_0, idx_1, edge_indexes, edge_attributes)
    d_02 = get_distance_attribute(idx_0, idx_2, edge_indexes, edge_attributes)
    d_12 = get_distance_attribute(idx_1, idx_2, edge_indexes, edge_attributes)
    
    # Reference the first three atoms
    x2 = (d_01**2 + d_02**2 - d_12**2) / (2 * d_01)
    y2 = np.sqrt(d_02**2 - x2**2)
    
    # Impose three particles at the beginning
    cartesian_positions = {
        idx_0: [0,    0,  0],
        idx_1: [d_01, 0,  0],
        idx_2: [x2,   y2, 0]
    }
    
    all_idxs = np.delete(np.arange(total_particles),
                         [idx_0, idx_1, idx_2])
    
    highest_n_explored = np.min(all_idxs)
    
    # Initialized to 3 for three connections
    n_connected_dict = {
        0: [],
        1: [],
        2: [],
        3: [highest_n_explored]
    }
    
    while True:  # Goes until all particles have been studied
        # Using a first-one-first-out approach
        idx = n_connected_dict[3][0]
        
        # Updated highest_n_explored with idx
        if idx > highest_n_explored:  # This allows tracking all particles
            highest_n_explored = idx
        
        n_connected, idx_connected = get_n_connected(idx, cartesian_positions, edge_indexes, edge_attributes)
        
        # Set idx in cartesian_positions or else add it to n_connected_dict
        if n_connected >= 3:
            # Extract the cartesian coordinates of idx
            temp_position = find_valid_reference(n_connected, idx_connected, edge_indexes, edge_attributes)
            
            # Check if there are enough particles able to locate idx (images make this step more difficult
            if temp_position is not None:
                # Generate temporal dictionary with the cartesian coordinates
                temp_dict = {
                    idx: temp_position
                }
                
                # Update general dictionary with cartesian coordinates
                cartesian_positions.update(temp_dict)
                
                # Now that cartesian_positions has been increased, n_connected_dict is updated
                n_connected_dict = update_n_connected_dict(n_connected_dict, idx, edge_indexes, edge_attributes)
            else:
                # Make idx wait for new connections to appear
                n_connected_dict[2].append(idx)
            
            # Remove idx from 3_connected_dict
            n_connected_dict[3].remove(idx)
        else:
            # n_connected_dict is updated adding idx where it belongs to
            n_connected_dict[n_connected].append(idx)
        
        # Check if all partciles have been already explored
        if highest_n_explored == total_particles:
            break
        
        # If not, if 3_connected_dict is finished, we add a new particle to be explored
        if not len(n_connected_dict[3]):
            n_connected_dict[3].append(highest_n_explored)
    
    return cartesian_positions


def update_n_connected_dict(n_connected_dict, idx_0, edge_indexes, edge_attributes):
    for i in np.arange(3+1)[::-1]:  # i = {3, 2, 1, 0}
        for idx_t in n_connected_dict[i]:
            if get_distance_attribute(idx_0, idx_t, edge_indexes, edge_attributes) is not None:
                # Remove from current list
                n_connected_dict[i].remove(idx_t)
                
                if i < 3:  # Else there is no list
                    # Append to next list
                    n_connected_dict[i+1].append(idx_t)
    return n_connected_dict


def get_n_connected(idx_0, cartesian_positions, edge_indexes, edge_attributes):
    n_connected = 0
    idx_connected = []
    for idx_t in list(cartesian_positions.keys()):
        if get_distance_attribute(idx_0, idx_t, edge_indexes, edge_attributes) is not None:
            n_connected += 1
            idx_connected.append(idx_t)
    
    return n_connected, idx_connected


def lattice_vectors_from_cartesian_positions(graph, cartesian_positions):
    """Calculate lattice vectors as a regression problem given their cartesian positions.

    Args:
        graph               (torch_geometric.data.Data): The input graph containing edge indexes and attributes.
        cartesian_positions (dict):                      A dictionary of atom positions in the format [x, y, z] for each atomic index, in angstroms.

    Returns:
        3x3 np.ndarray: Simulation cell (lattice vectors).
    
    x + (alpha, beta, gamma) (a1, a2, a3) = x'
    """
    

    return lattice_vectors


def check_graph_validity(graph):
    """Check that the current graph describes a realistic material (positive interatomic distances, etc.).

    Args:
        graph (torch_geometric.data.Data): The input graph containing edge indexes and attributes.
        
    Raises:
        SystemExit: If the provided graph has negative or null interatomic distances.
    """
    
    if torch.any(graph.edge_attr <= 0):
        print('Invalid graph, atoms overlapping. Applying brute force :)')
        graph.edge_attr[graph.edge_attr <= 0] = 0
