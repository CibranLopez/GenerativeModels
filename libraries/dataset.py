import json
import numpy as np
import os 
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


def standardize_dataset(dataset_path, dest, transformation=None):
    """Standardizes a given dataset (both nodes features and edge attributes).
    Typically, a normal distribution is applied, although it be easily modified to apply other distributions.
    Check those graphs with finite attributes and retains labels accordingly.

    Currently: normal distribution.

    Parameters
    ----------
    dataset_path: str
        Path to the dataset. It must include the following files:
            dataset.pt: A list containing graph structures.
            labels.pt: A list containing labels for each graph.
            train_labels.txt: A list of labels for the training set.
            val_labels.txt: A list of labels for the validation set.
            test_labels.txt: A list of labels for the test set.

    dest: str
        Path to save the standardized dataset.
    """
    print("Loading original dataset...")
    dataset = torch.load(os.path.join(dataset_path, 'dataset.pt'), weights_only=False)
    labels  = torch.load(os.path.join(dataset_path, 'labels.pt'),  weights_only=False)

    print("Checking finite attributes...")
    # Clone the dataset and labels
    dataset_std = []
    labels_std  = []
    graph_idx = 0
    for graph, label in zip(dataset, labels):
        if check_finite_attributes(graph):
            dataset_std.append(graph.clone())
            labels_std.append(label)
        graph_idx += 1
        print(f"Graph {graph_idx} checked...")


    # Apply the transformation to the edge attributes
    if transformation == 'inverse-quadratic':
        print("Applying inverse quadratic transformation to edge attributes...")
        graph_idx = 0
        for data in dataset_std:
            data.edge_attr = 1 / data.edge_attr.pow(2)
            graph_idx += 1
            print(f"Graph {graph_idx} transformed...")

    # Load dataset subsets
    print("Reading labels...")
    train_labels = np.genfromtxt(os.path.join(dataset_path, 'train_labels.txt'), dtype='str').tolist()
    val_labels   = np.genfromtxt(os.path.join(dataset_path, 'val_labels.txt'),   dtype='str').tolist()
    test_labels  = np.genfromtxt(os.path.join(dataset_path, 'test_labels.txt'),  dtype='str').tolist()

    print("Getting subsets...")
    train_dataset_std = get_datasets(train_labels, labels_std, dataset_std)
    val_dataset_std = get_datasets(val_labels,  labels_std, dataset_std)
    test_dataset_std  = get_datasets(test_labels,  labels_std , dataset_std)

    # Create the mean and standard deviation for the training set
    n_graphs = len(train_dataset_std)
    n_features = train_dataset_std[0].num_node_features
    n_y = train_dataset_std[0].y.shape[0]
    
    # Transform edge attributes
    print("Transforming edge attributes...")
    """
    target_mean = torch.zeros(n_y)
    for target_index in range(n_y):
        target_mean[target_index] = sum([data.y[target_index] for data in train_dataset_std]) / n_graphs


    target_std = torch.zeros(n_y)
    for target_index in range(n_y):
        target_std[target_index] = torch.sqrt(sum([(data.y[target_index] - target_mean[target_index]).pow(2).sum() for data in train_dataset_std]) / (n_graphs * (n_graphs - 1)))
    """
    # Compute total mean for target attributes (y)
    target_sum = torch.zeros(n_y)  # Accumulator for target sums
    total_target_count = 0  # Total count of target entries

    for data in train_dataset_std:
        target_sum += data.y.sum(dim=0)  # Accumulate target values across all graphs
        total_target_count += data.y.shape[0]  # Count the number of entries in y

    target_mean = target_sum / total_target_count  # Compute global mean for each target attribute


    # Compute total standard deviation for target attributes (y)
    target_sum = torch.zeros(n_y)  # Accumulator for target sums
    total_target_count = 0  # Total count of target entries

    for data in train_dataset_std:
        target_sum += data.y.sum(dim=0)  # Accumulate target values across all graphs
        total_target_count += data.y.shape[0]  # Count the number of entries in y

    target_mean = target_sum / total_target_count  # Compute global mean for each target attribute

    # Compute variance and standard deviation
    target_variance = torch.zeros(n_y)
    for data in train_dataset_std:
        target_variance += ((data.y - target_mean).pow(2)).sum(dim=0)  # Accumulate squared deviations

    target_variance /= total_target_count  # Divide by total count to get variance
    target_std = torch.sqrt(target_variance)  # Compute standard deviation

    """    
    edge_mean = sum([data.edge_attr.mean() for data in train_dataset_std]) / n_graphs
    edge_std = torch.sqrt(sum([(data.edge_attr - edge_mean).pow(2).sum() for data in train_dataset_std]) / (n_graphs * (n_graphs - 1)))
    """

    edge_sum = 0
    total_edge_count = 0

    for data in train_dataset_std:
        edge_sum += data.edge_attr.sum()  # Sum all edge attributes
        total_edge_count += data.edge_attr.numel()  # Add the number of elements in edge_attr

    edge_mean = edge_sum / total_edge_count  # Compute the total mean

    edge_variance = sum([(data.edge_attr - edge_mean).pow(2).sum() for data in train_dataset_std]) / total_edge_count
    edge_std = torch.sqrt(edge_variance)  # Compute standard deviation
    
    # In case we want to increase the values of the normalization
    scale = torch.tensor(1e0)

    target_factor = target_std / scale
    edge_factor   = edge_std   / scale

    for dataset in [train_dataset_std, val_dataset_std, test_dataset_std]:
        print(f"Transforming {len(dataset)} graphs...")
        for data in dataset:
            data.y         = (data.y         - target_mean) / target_factor
            data.edge_attr = (data.edge_attr - edge_mean)   / edge_factor

    """
    # Transform node attributes
    feat_mean = torch.zeros(n_features)
    feat_std  = torch.zeros(n_features)
    print("Transforming node attributes...")
    for feat_index in range(n_features):
        # Compute mean
        temp_feat_mean = sum([data.x[:, feat_index].mean() for data in train_dataset_std]) / n_graphs
        
        # Compute standard deviations
        temp_feat_std = torch.sqrt(sum([(data.x[:, feat_index] - temp_feat_mean).pow(2).sum() for data in train_dataset_std]) / (n_graphs * (n_graphs - 1)))

        # Update normalized values into the database
        for dataset in [train_dataset_std, val_dataset_std, test_dataset_std]:
            print(f"Transforming {len(dataset)} graphs for feature {feat_index}...")
            for data in dataset:
                data.x[:, feat_index] = (data.x[:, feat_index] - temp_feat_mean) * scale / temp_feat_std
        
        # Append corresponing values for saving
        feat_mean[feat_index] = temp_feat_mean
        feat_std[feat_index]  = temp_feat_std
    """
    print("Transforming node attributes...")
    # Initialize accumulators for global mean and standard deviation
    feat_sum = torch.zeros(n_features)  # Accumulator for feature sums
    total_feat_count = torch.zeros(n_features)  # Accumulator for the total count of each feature

    # Compute global mean
    for data in train_dataset_std:
        feat_sum += data.x.sum(dim=0)  # Sum node features for each feature index
        total_feat_count += torch.tensor([data.x[:, i].numel() for i in range(n_features)])  # Count the number of nodes for each feature

    feat_mean = feat_sum / total_feat_count  # Compute global mean for each feature

    # Compute global standard deviation
    feat_variance = torch.zeros(n_features)
    for data in train_dataset_std:
        feat_variance += ((data.x - feat_mean).pow(2)).sum(dim=0)  # Accumulate squared deviations for each feature

    feat_variance /= total_feat_count  # Divide by total count for variance
    feat_std = torch.sqrt(feat_variance)  # Compute standard deviation

    # Normalize features for all datasets
    for dataset in [train_dataset_std, val_dataset_std, test_dataset_std]:
        print(f"Transforming {len(dataset)} graphs...")
        for data in dataset:
            data.x = (data.x - feat_mean) / feat_std  # Apply normalization


   

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
    
    # Join the subsets and save the standardized dataset
    dataset_std = train_dataset_std + val_dataset_std + test_dataset_std
    torch.save(dataset_std, os.path.join(dest, 'standardized_dataset.pt'))
    torch.save(labels_std,  os.path.join(dest, 'standardized_labels.pt'))
    
    # Convert torch tensors to numpy arrays
    numpy_dict = {}
    for key, value in dataset_parameters.items():
        try:
            numpy_dict[key] = value.cpu().numpy().tolist()
        except:
            numpy_dict[key] = value

    # Dump the dictionary with numpy arrays to a JSON file
    with open(os.path.join(dest, "standardized_parameters.json"), 'w') as json_file:
        json.dump(numpy_dict, json_file)


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

    subset_labels = np.array(subset_labels)
    dataset_labels = np.array(dataset_labels)

    # Extract the first word of each dataset_label
    dataset_first_words = np.array([label.split()[0] for label in dataset_labels])

    # Find matches
    matches = np.isin(dataset_first_words, subset_labels)

    # Get the indices
    dataset_idxs = np.where(matches)[0]

    # Return the subset of the dataset
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
