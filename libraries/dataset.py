import numpy as np
import torch

# Checking if pytorch can run in GPU, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def check_finite_attributes(data):
    """
    Checks if all node and edge attributes in the graph are finite (i.e., not NaN, inf, or -inf).

    Args:
        data: A graph object containing node attributes (`data.x`) and edge attributes (`data.edge_attr`).

    Returns:
        bool: 
            - True if all node and edge attributes are finite.
            - False if any node or edge attributes are NaN, inf, or -inf.
    """
    # Check node attributes
    if not torch.any(torch.isfinite(data.x)):
        return False

    # Check edge attributes
    if not torch.any(torch.isfinite(data.edge_attr)):
        return False
    return True


def standardize_dataset(dataset, labels, transformation=None):
    """Standardizes a given dataset (both nodes features and edge attributes).
    Typically, a normal distribution is applied, although it be easily modified to apply other distributions.
    Check those graphs with finite attributes and retains labels accordingly.

    Currently: normal distribution.

    Args:
        dataset        (list): List containing graph structures.
        labels         (list): List containing graph labels.
        transformation (str):  Type of transformation strategy for edge attributes (None, 'inverse-quadratic').

    Returns:
        Tuple: A tuple containing the normalized dataset and parameters needed to re-scale predicted properties.
            - dataset_std        (list): Normalized dataset.
            - labels_std         (list): Labels from valid graphs.
            - dataset_parameters (dict): Parameters needed to re-scale predicted properties from the dataset.
    """

    # Clone the dataset and labels
    dataset_std = []
    labels_std  = []
    for graph, label in zip(dataset, labels):
        if check_finite_attributes(graph):
            dataset_std.append(graph.clone())
            labels_std.append(label)

    # Number of graphs
    n_graphs = len(dataset_std)
    
    # Number of features per node
    n_features = dataset_std[0].num_node_features
    
    # Number of features per graph
    n_y = dataset_std[0].y.shape[0]
    
    # Check if non-linear standardization
    if transformation == 'inverse-quadratic':
        for data in dataset_std:
            data.edge_attr = 1 / data.edge_attr.pow(2)

    # Compute means
    target_mean = torch.zeros(n_y)
    for target_index in range(n_y):
        target_mean[target_index] = sum([data.y[target_index] for data in dataset_std]) / n_graphs
    
    edge_mean = sum([data.edge_attr.mean() for data in dataset_std]) / n_graphs
    
    # Compute standard deviations
    target_std = torch.zeros(n_y)
    for target_index in range(n_y):
        target_std[target_index] = torch.sqrt(sum([(data.y[target_index] - target_mean[target_index]).pow(2).sum() for data in dataset_std]) / (n_graphs * (n_graphs - 1)))
    
    edge_std = torch.sqrt(sum([(data.edge_attr - edge_mean).pow(2).sum() for data in dataset_std]) / (n_graphs * (n_graphs - 1)))
    
    # In case we want to increase the values of the normalization
    scale = torch.tensor(1e0)

    target_factor = target_std / scale
    edge_factor   = edge_std   / scale

    # Update normalized values into the database
    for data in dataset_std:
        data.y         = (data.y         - target_mean) / target_factor
        data.edge_attr = (data.edge_attr - edge_mean)   / edge_factor

    # Same for the node features
    feat_mean = torch.zeros(n_features)
    feat_std  = torch.zeros(n_features)
    for feat_index in range(n_features):
        # Compute mean
        temp_feat_mean = sum([data.x[:, feat_index].mean() for data in dataset_std]) / n_graphs
        
        # Compute standard deviations
        temp_feat_std = torch.sqrt(sum([(data.x[:, feat_index] - temp_feat_mean).pow(2).sum() for data in dataset_std]) / (n_graphs * (n_graphs - 1)))

        # Update normalized values into the database
        for data in dataset_std:
            data.x[:, feat_index] = (data.x[:, feat_index] - temp_feat_mean) * scale / temp_feat_std
        
        # Append corresponing values for saving
        feat_mean[feat_index] = temp_feat_mean
        feat_std[feat_index]  = temp_feat_std

    # Create and save as a dictionary
    dataset_parameters = {
        'transformation': transformation,
        'target_mean':    target_mean,
        'feat_mean':      feat_mean,
        'edge_mean':      edge_mean,
        'target_std':     target_std,
        'edge_std':       edge_std,
        'feat_std':       feat_std,
        'scale':          scale
    }
    return dataset_std, labels_std, dataset_parameters


def revert_standardize_dataset(dataset, dataset_parameters):
    """De-standardizes a given dataset (both nodes features and edge attributes).
    Typically, a normal distribution is applied, although it be easily modified to apply other distributions.

    Currently: normal distribution.

    Args:
        dataset            (list): List containing graph structures.
        dataset_parameters (dict): Parameters needed to re-scale predicted properties from the dataset.

    Returns:
        dataset_rstd (list): De-normalized dataset.
    """
    
    # Clone the dataset (using a list comprehension)
    dataset_rstd = [graph.clone() for graph in dataset]
    
    edge_factor = dataset_parameters['edge_std'] / dataset_parameters['scale']

    # Update normalized values into the database
    for data in dataset_rstd:
        data.edge_attr = data.edge_attr * edge_factor + dataset_parameters['edge_mean']

    # Same for the node features
    for feat_index in range(dataset_rstd[0].num_node_features):
        for data in dataset_rstd:
            data.x[:, feat_index] = data.x[:, feat_index] * dataset_parameters['feat_std'][feat_index] / dataset_parameters['scale'] + dataset_parameters['feat_mean'][feat_index]

    return dataset_rstd


def get_datasets(subset_labels, dataset_labels, dataset):
    """Get datasets filtered, non-ordered by labels.

    Args:
        subset_labels  (list): List of labels to filter by.
        dataset_labels (list): List of material labels.
        dataset        (list): List of data elements.

    Returns:
        list: Filtered dataset containing elements corresponding to the specified labels (not ordered).
    """

    subset_labels  = np.array(subset_labels)
    dataset_labels = np.array(dataset_labels)
    
    dataset_idxs = []
    for dataset_idx, dataset_label in enumerate(dataset_labels):
        for subset_idx, subset_label in enumerate(subset_labels):
            if dataset_label.split()[0] == subset_label:
                dataset_idxs.append(dataset_idx)
                subset_labels = np.delete(subset_labels, subset_idx)
        if not len(subset_labels):
            break
    return [dataset[idx] for idx in dataset_idxs]


def check_extend_POSCAR(structure, minimum_lattice_vector):
    """Check that POSCAR cell is large enough, otherwise extend it in the direction.
    A new POSCAR is saved, replacing previous one, which is copied to POSCAR_ini.

    Args:
        structure              (pymatgen Structure object): Structure from which the graph is to be generated.
        minimum_lattice_vector (float):                     Minimum length of lattice vectors to be able to perform convolutions.
    """

    # Get necessary transformation for POSCAR to have valid lengths
    replication_factor = np.ceil(minimum_lattice_vector / np.linalg.norm(structure.lattice.matrix, axis=1))

    if np.all(replication_factor > 0):
        structure.make_supercell(replication_factor)
    return structure
