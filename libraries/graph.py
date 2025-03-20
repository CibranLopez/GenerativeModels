import numpy as np
import torch
import itertools
import sys

from pymatgen.core.structure import Structure
from scipy.spatial           import Voronoi
from rdkit                   import Chem

# Checking if pytorch can run in GPU, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_all_linked_tessellation(
        atomic_data,
        structure
):
    """Gets the distances by pairs of particles, considering images with periodic boundary conditions (PBC).

    Args:
        atomic_data        (dict):                      A dictionary with all node features.
        structure          (pymatgen Structure object): Structure from which the graph is to be generated
        distance_threshold (float, optional):           The distance threshold for edge creation (default is 6).

    Returns:
        nodes      (list): A tensor containing node attributes.
        edges      (list): A tensor containing edge indices.
        attributes (list): A tensor containing edge attributes (distances).
    """

    # Extract direct positions, composition and concentration as lists
    positions     = np.array([site.frac_coords for site in structure.sites])
    composition   = [element.symbol for element in structure.composition.elements]
    concentration = np.array([sum(site.species_string == element for site in structure.sites) for element in composition])

    # Counting number of particles
    total_particles = np.sum(concentration)

    # Generating graph structure, getting particle types
    particle_types = []
    for i in range(len(composition)):
        particle_types += [i] * concentration[i]

    # Adding nodes and edges.
    nodes = []
    edges = []
    attributes = []
    for index_0 in range(total_particles):
        # Get particle type (index of type wrt composition in POSCAR)
        particle_type = particle_types[index_0]

        # Name of the current species
        species_name = composition[particle_type]

        # Adding the nodes (mass, charge, electronegativity and ionization energies)
        nodes.append([atomic_data[species_name]['atomic_mass'],
                      atomic_data[species_name]['charge'],
                      atomic_data[species_name]['electronegativity'],
                      atomic_data[species_name]['ionization_energy']])

        # Get the initial position
        position_0 = positions[index_0]
        position_cartesian_0 = np.dot(position_0, structure.lattice.matrix)

        # Explore images of all particles in the system
        # Starting on index_0, thus exploring possible images with itself (except for i,j,k=0, exact same particle)
        for index_i in np.arange(index_0, total_particles):
            # Get the initial position
            position_i = positions[index_i]

            # Move to the corresponding image and convert to cartesian distances
            position_cartesian_i = np.dot(position_i, structure.lattice.matrix)

            # New distance as Euclidean distance between both reference and new image particle
            distance = np.linalg.norm([position_cartesian_0 - position_cartesian_i])

            # Append this point as an edge connection to particle 0
            edges.append([index_0, index_i])
            attributes.append([distance])
    return nodes, edges, attributes


def get_voronoi_tessellation(
        atomic_data,
        temp_structure,
        periodicity
):
    """
    Get the Voronoi nodes of a structure.
    Templated from the TopographyAnalyzer class, added to pymatgen.analysis.defects.utils by Yiming Chen, but now deleted.
    Modified to map down to primitive, do Voronoi analysis, then map back to original supercell; much more efficient.
    See commit 8b78474 'Generative models (basic example).ipynb'.

    Args:
        atomic_data    (dict):                      A dictionary with all node features.
        temp_structure (pymatgen Structure object): Structure from which the graph is to be generated.
        periodicity    (bool):                      Whether to consider periodicity of the structure.
    """
    
    # Map all sites to the unit cell; 0 ≤ xyz < 1
    structure = Structure.from_sites(temp_structure, to_unit_cell=True)

    # Get Voronoi nodes in primitive structure and then map back to the
    # supercell
    prim_structure = structure.get_primitive_structure()

    # Get all atom coords in a supercell of the structure because
    # Voronoi polyhedra can extend beyond the standard unit cell
    coords = []
    if periodicity: cell_range = list(range(-1, 2))  # Periodicity
    else:           cell_range = [0]  # No periodicity
    for shift in itertools.product(cell_range, cell_range, cell_range):
        for site in prim_structure.sites:
            shifted = site.frac_coords + shift
            coords.append(prim_structure.lattice.get_cartesian_coords(shifted))

    # Voronoi tessellation
    voro = Voronoi(coords)

    tol = 1e-6
    new_ridge_points = []
    for atoms in voro.ridge_points:  # Atoms are indexes referred to coords
        # Dictionary for storing information about each atom
        atoms_info = {}

        new_atoms = []
        # Check if any of those atoms belong to the unitcell
        for atom_idx in range(2):
            atom = atoms[atom_idx]

            # Direct coordinates from supercell referenced to the primitive cell
            frac_coords = prim_structure.lattice.get_fractional_coords(coords[atom])

            is_atom_inside = True
            frac_coords_uc = frac_coords
            if not np.all([-tol <= coord < 1 + tol for coord in frac_coords]):
                # atom_x is not inside
                is_atom_inside = False

                # Apply periodic boundary conditions
                while np.any(frac_coords_uc > 1): frac_coords_uc[np.where(frac_coords_uc > 1)] -= 1
                while np.any(frac_coords_uc < 0): frac_coords_uc[np.where(frac_coords_uc < 0)] += 1

            # Obtain mapping to index in unit cell
            uc_idx = np.argmin(np.linalg.norm(prim_structure.frac_coords - frac_coords_uc, axis=1))
            
            if is_atom_inside:
                new_atoms.append(str(uc_idx))
            else:
                new_atoms.append('-'+str(uc_idx))
        
        distance = np.linalg.norm(coords[atoms[1]] - coords[atoms[0]])
        new_atoms.append(distance)
        new_atoms.append(atoms[0])
        new_atoms.append(atoms[1])
        
        new_ridge_points.append(new_atoms)
    
    # Delete those edges which only contain images
    to_delete = []
    for k in range(len(new_ridge_points)):
        pair = new_ridge_points[k][:2]
        if (pair[0][0] == '-') and (pair[1][0] == '-'):
            to_delete.append(k)
    new_ridge_points = np.delete(new_ridge_points, to_delete, axis=0)
    
    edges      = []
    attributes = []
    for idx_i in range(temp_structure.num_sites):
        for idx_j in np.arange(idx_i+1, temp_structure.num_sites):
            to_delete = []
            for k in range(len(new_ridge_points)):
                pair = new_ridge_points[k, :2]
                dist = new_ridge_points[k, 2]
                
                if np.any(pair == str(idx_i)):  # Real for idx_i
                    if pair[0][0] == '-': pair[0] = pair[0][1:]
                    if pair[1][0] == '-': pair[1] = pair[1][1:]
                    
                    if np.any(pair == str(idx_j)):  # Real or image for idx_j
                        edges.append(np.array(pair, dtype=int))
                        attributes.append(float(dist))
                        to_delete.append(k)

            # Delete these added edges, which are no longed needed
            new_ridge_points = np.delete(new_ridge_points, to_delete, axis=0)

    edges      = np.array(edges)
    attributes = np.array(attributes)

    # Generate nodes from all atoms in structure
    nodes = []
    for idx in range(structure.num_sites):
        # Get species type
        species_name = str(structure[idx].species)[:-1]

        # Get node info
        # Loading the node (mass, charge, electronegativity and ionization energy)
        nodes.append([atomic_data[species_name]['atomic_mass'],
                      atomic_data[species_name]['charge'],
                      atomic_data[species_name]['electronegativity'],
                      atomic_data[species_name]['ionization_energy']])
    return nodes, edges, attributes


def get_sphere_images_tessellation(
        atomic_data,
        structure,
        distance_threshold=6
):
    """Gets the distances by pairs of particles, considering images with periodic boundary conditions (PBC).

    Args:
        atomic_data        (dict):                      A dictionary with all node features.
        structure          (pymatgen Structure object): Structure from which the graph is to be generated
        distance_threshold (float, optional):           The distance threshold for edge creation (default is 6).

    Returns:
        nodes      (list): A tensor containing node attributes.
        edges      (list): A tensor containing edge indices.
        attributes (list): A tensor containing edge attributes (distances).
    """

    # Extract direct positions, composition and concentration as lists
    positions     = np.array([site.frac_coords for site in structure.sites])
    composition   = [element.symbol for element in structure.composition.elements]
    concentration = np.array([sum(site.species_string == element for site in structure.sites) for element in composition])

    # Counting number of particles
    total_particles = np.sum(concentration)

    # Generating graph structure, getting particle types
    particle_types = []
    for i in range(len(composition)):
        particle_types += [i] * concentration[i]

    # Adding nodes and edges.
    nodes = []
    edges = []
    attributes = []
    for index_0 in range(total_particles):
        # Get particle type (index of type wrt composition in POSCAR)
        particle_type = particle_types[index_0]

        # Name of the current species
        species_name = composition[particle_type]

        # Adding the nodes (mass, charge, electronegativity and ionization energies)
        nodes.append([atomic_data[species_name]['atomic_mass'],
                      atomic_data[species_name]['charge'],
                      atomic_data[species_name]['electronegativity'],
                      atomic_data[species_name]['ionization_energy']])

        # Get the initial position
        position_0 = positions[index_0]
        position_cartesian_0 = np.dot(position_0, structure.lattice.matrix)

        # Explore images of all particles in the system
        # Starting on index_0, thus exploring possible images with itself (except for i,j,k=0, exact same particle)
        for index_i in np.arange(index_0, total_particles):
            # Get the initial position
            position_i = positions[index_i]

            reference_distance_i = np.nan  # So it outputs False when first compared with another distance
            i = 0
            alpha_i = 1
            while True:
                minimum_distance_i   = np.nan
                reference_distance_j = np.nan
                j = 0
                alpha_j = 1
                while True:
                    minimum_distance_j   = np.nan
                    reference_distance_k = np.nan
                    k = 0
                    alpha_k = 1
                    while True:
                        # Move to the corresponding image and convert to cartesian distances
                        position_cartesian_i = np.dot(position_i + [i, j, k], structure.lattice.matrix)

                        # New distance as Euclidean distance between both reference and new image particle
                        new_distance = np.linalg.norm([position_cartesian_0 - position_cartesian_i])

                        # Condition that remove exact same particle
                        same_index_condition     = (index_0 == index_i)
                        all_index_null_condition = np.all([i, j, k] == [0]*3)
                        same_particle_condition  = (same_index_condition and all_index_null_condition)

                        # Applying threshold to images
                        if (new_distance <= distance_threshold) and not same_particle_condition:
                            # Append this point as a edge connection to particle 0
                            edges.append([index_0, index_i])
                            attributes.append([new_distance])

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
    return nodes, edges, attributes


def get_molecule_tessellation(
        atomic_data,
        smiles
):
    """Extracts graph information from SMILES codification of a molecule.

    Args:
        atomic_data (dict): A dictionary with all node features.
        smiles      (str): SMILES string codifying a molecule.

    Returns:
        nodes      (list): A tensor containing node attributes.
        edges      (list): A tensor containing edge indices.
        attributes (list): A tensor containing edge attributes (distances).
    """

    # Generate the molecule from the SMILES string
    mol = Chem.MolFromSmiles(smiles)

    # Get edge attributes (bond types)
    edges      = []
    attributes = []
    for i in range(mol.GetNumAtoms()):
        for j in range(i+1, mol.GetNumAtoms()):
            bond = mol.GetBondBetweenAtoms(i, j)
            if bond is not None:
                bond_type = bond.GetBondTypeAsDouble()
                edges.append([i, j])
                attributes.append(bond_type)

    # Generate node features
    nodes = []
    for atom in mol.GetAtoms():
        species_name = atom.GetSymbol()
        nodes.append([atomic_data[species_name]['atomic_mass'],
                      atomic_data[species_name]['charge'],
                      atomic_data[species_name]['electronegativity'],
                      atomic_data[species_name]['ionization_energy']])
    return nodes, edges, attributes


def graph_POSCAR_encoding(
        structure,
        encoding_type='voronoi',
        distance_threshold=6,
        periodicity=True
):
    """Generates a graph parameters from a POSCAR.
    There are the following implementations:
        1. Voronoi tessellation.
        2. All particles inside a sphere of radius distance_threshold.
        3. Filled space given a cubic box of dimension [0-Lx, 0-Ly, 0-Lz] considering all necessary images.
           It links every particle with the rest for the given set of nodes and edges.

    Args:
        structure          (pymatgen Structure object): Structure from which the graph is to be generated.
        encoding_type      (str):    Framework used for encoding the structure.
        distance_threshold (float):  Distance threshold for sphere-images tessellation.
        periodicity        (bool):   Whether or not to consider periodicity of the structure.
    Returns:
        nodes      (torch tensor): Generated nodes with corresponding features.
        edges      (torch tensor): Generated connections between nodes.
        attributes (torch tensor): Corresponding weights of the generated connections.
    """

    # Loading dictionary of atomic masses
    atomic_data = {}
    with open('input/atomic_masses.dat', 'r') as atomic_data_file:
        for line in atomic_data_file:
            key, atomic_mass, charge, electronegativity, ionization_energy = line.split()
            atomic_data[key] = {
                'atomic_mass':       float(atomic_mass) if atomic_mass != 'None' else None,
                'charge':            int(charge) if charge != 'None' else None,
                'electronegativity': float(electronegativity) if electronegativity != 'None' else None,
                'ionization_energy': float(ionization_energy) if ionization_energy != 'None' else None
            }

    if encoding_type == 'voronoi':
        # Get edges and attributes for the corresponding tessellation
        nodes, edges, attributes = get_voronoi_tessellation(atomic_data,
                                                            structure,
                                                            periodicity)

    elif encoding_type == 'sphere-images':
        # Get edges and attributes for the corresponding tessellation
        nodes, edges, attributes = get_sphere_images_tessellation(atomic_data,
                                                                  structure,
                                                                  distance_threshold=distance_threshold)

    elif encoding_type == 'all-linked':
        # Get edges and attributes for the corresponding tessellation
        nodes, edges, attributes = get_all_linked_tessellation(atomic_data,
                                                               structure)

    elif encoding_type == 'molecule':
        # Get edges and attributes for the corresponding tessellation
        nodes, edges, attributes = get_molecule_tessellation(atomic_data,
                                                             structure)

    else:
        sys.exit('Error: encoding type not available.')

    # Convert to torch tensors and return
    nodes      = torch.tensor(nodes,      dtype=torch.float)
    edges      = torch.tensor(edges,      dtype=torch.long)
    attributes = torch.tensor(attributes, dtype=torch.float)
    return nodes, edges, attributes


def find_closest_key(
        dictionary,
        target_array
):
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


def discretize_graph(
        graph
):
    """Convert the graph's continuous node embeddings to the closest valid embeddings based on the periodic table.

    Args:
        graph (torch_geometric.data.Data): The initial graph structure with continuous node embeddings.

    Returns:
        new_graph (torch_geometric.data.Data): The modified graph with the most valid node embeddings based on the periodic table.
    """

    # Clone the input graph to preserve the original structure
    new_graph = graph.clone()

    # Detach embeddings for the graph nodes
    data_embeddings = new_graph.x.detach()

    # Load the dictionary of available embeddings for atoms
    available_embeddings = {}
    with open('input/atomic_masses.dat', 'r') as atomic_masses_file:
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


def composition_concentration_from_keys(
        keys,
        positions
):
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


def POSCAR_graph_encoding(
        graph,
        lattice_vectors,
        file_name='POSCAR',
        POSCAR_name=None,
        POSCAR_directory='./'
):
    """Encode a graph into a POSCAR (VASP input) file format.

    Args:ººº
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
    with open('input/atomic_masses.dat', 'r') as atomic_masses_file:
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


def check_graph_validity(
        graph
):
    """Check that the current graph describes a realistic material (positive interatomic distances, etc.).

    Args:
        graph (torch_geometric.data.Data): The input graph containing edge indexes and attributes.
        
    Raises:
        SystemExit: If the provided graph has negative or null interatomic distances.
    """
    
    if torch.any(graph.edge_attr <= 0):
        print('Invalid graph, atoms overlapping. Applying brute force :)')
        graph.edge_attr[graph.edge_attr <= 0] = 0
