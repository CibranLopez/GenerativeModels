import torch

# Checking if pytorch can run in GPU, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def standardize_dataset(dataset):
    """Standardizes a given dataset (both nodes features and edge attributes).
    Typically, a normal distribution is applied, although it be easily modified to apply other distributions.

    Currently: normal distribution.

    Args:
        dataset (list): List containing graph structures.

    Returns:
        Tuple: A tuple containing the normalized dataset and parameters needed to re-scale predicted properties.
            - dataset_std        (list): Normalized dataset.
            - dataset_parameters (dict): Parameters needed to re-scale predicted properties from the dataset.
    """

    # Clone the dataset (using a list comprehension)
    dataset_std = [graph.clone() for graph in dataset]

    # Number of graphs
    n_graphs   = len(dataset_std)
    n_features = dataset_std[0].num_node_features
    
    # Compute means
    target_mean = sum([data.y.mean()         for data in dataset_std]) / n_graphs
    edge_mean   = sum([data.edge_attr.mean() for data in dataset_std]) / n_graphs
    
    # Compute standard deviations
    target_std = torch.sqrt(sum([(data.y         - target_mean).pow(2).sum() for data in dataset_std]) / (n_graphs * (n_graphs - 1)))
    edge_std   = torch.sqrt(sum([(data.edge_attr -   edge_mean).pow(2).sum() for data in dataset_std]) / (n_graphs * (n_graphs - 1)))
    
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
        'target_mean': target_mean,
        'feat_mean':   feat_mean,
        'edge_mean':   edge_mean,
        'target_std':  target_std,
        'edge_std':    edge_std,
        'feat_std':    feat_std,
        'scale':       scale
    }
    return dataset_std, dataset_parameters


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
        data.edge_attr = data.edge_attr * dataset_parameters['edge_factor'] + dataset_parameters['edge_mean']

    # Same for the node features
    for feat_index in range(dataset_rstd[0].num_node_features):
        for data in dataset_rstd:
            data.x[:, feat_index] = data.x[:, feat_index] * dataset_parameters['feat_std'][feat_index] / dataset_parameters['scale'] + dataset_parameters['feat_mean'][feat_index]

    return dataset_rstd
