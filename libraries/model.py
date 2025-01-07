import logging


import numpy               as np
import matplotlib.pyplot   as plt
import torch.nn.functional as F
import torch.nn            as nn
import networkx            as nx
import os 
import pandas as pd
import time
import torch
import sys
import yaml

from torch_geometric.data          import Data, Batch
from torch.nn                      import Linear
from torch_geometric.nn            import GraphConv, GraphNorm

from torch_geometric.utils.convert import to_networkx

# Checking if pytorch can run in GPU, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_alpha_t(t_step, n_t_steps, alpha_decay):
    """Defines constant alpha at time-step t, given a parameter s < 0.5 (else alpha increases).
    
    \\alpha (t) = (1 - 2 s) \\left( 1 - \\left( \\frac{t}{T} \\right)^2 \\right) + s

    where alpha_-1 = 1.

    Args:
        t_step      (int):   time step (of diffusion or denoising) in which alpha is required.
        n_t_steps   (int):   total number of steps.
        alpha_decay (float): parameter which controls the decay of alpha with t.

    Returns:
        alpha (float): parameter which controls how much signal is retained.
    """
    
    if t_step < 0:
        return torch.tensor(1, dtype=torch.int, device=device)
    return (1 - 2 * alpha_decay) * (1 - (t_step / n_t_steps)**2) + alpha_decay


def get_sigma_t(t_step, n_t_steps, alpha_decay):
    """Defines constant sigma at time-step t, given alpha at t.

    \\sigma (t) = \\sqrt( 1 - \\alpha^2_t )

    Args:
        t_step      (int):   time step (of diffusion or denoising) in which alpha is required.
        n_t_steps   (int):   total number of steps.
        alpha_decay (float): parameter which controls the decay of alpha with t.

    Returns:
        sigma (float): parameter which controls how much noise is added.
    """

    alpha_t = get_alpha_t(t_step, n_t_steps, alpha_decay)
    return torch.sqrt(1 - alpha_t**2)


def get_alpha_t_s(t, s, n_t_steps, alpha_decay):
    """Computes sigma_t over sigma_s.

    \\alpha (t, s) = \\alpha_t / \\alpha_s

    Args:
        t           (int):   time step.
        s           (int):   time step.
        n_t_steps   (int):   total number of steps.
        alpha_decay (float): parameter which controls the decay of alpha with t.

    Returns:
        alpha_t over alpha_s (float).
    """

    alpha_t = get_alpha_t(t, n_t_steps, alpha_decay)
    alpha_s = get_alpha_t(s, n_t_steps, alpha_decay)

    return alpha_t / alpha_s


def get_sigma_t_s(t, s, n_t_steps, alpha_decay):
    """Computes sigma_t over sigma_s.

    \\sigma (t, s) = \\sqrt( \\sigma^2_t - \\alpha^2_{t,s} \\sigma^2_s )

    Args:
        t           (int):   time step.
        s           (int):   time step.
        n_t_steps   (int):   total number of steps.
        alpha_decay (float): parameter which controls the decay of alpha with t.

    Returns:
        sigma_t over sigma_s (float).
    """

    alpha_t_s = get_alpha_t_s(t, s, n_t_steps, alpha_decay)
    sigma_t   = get_sigma_t(t, n_t_steps, alpha_decay)
    sigma_s   = get_sigma_t(s, n_t_steps, alpha_decay)
    return torch.sqrt(sigma_t**2 - alpha_t_s**2 * sigma_s**2)


def get_sigma_t_to_s(t, s, n_t_steps, alpha_decay):
    """Computes sigma_t over sigma_s.

    \\sigma (t, s) = \\sqrt( \\sigma^2_t - \\alpha^2_{t,s} \\sigma^2_s )

    Args:
        t           (int):   time step.
        s           (int):   time step.
        n_t_steps   (int):   total number of steps.
        alpha_decay (float): parameter which controls the decay of alpha with t.

    Returns:
        sigma_t over sigma_s (float).
    """

    sigma_t   = get_sigma_t(t, n_t_steps, alpha_decay)
    sigma_s   = get_sigma_t(s, n_t_steps, alpha_decay)
    sigma_t_s = get_sigma_t_s(t, s, n_t_steps, alpha_decay)

    return sigma_t_s * sigma_s / sigma_t


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
    
    #TODO: remove
    torch.manual_seed(12345)
    # Generate random node features
    x = torch.randn(n_nodes, n_features)
    torch.seed()

    # Get number of edges
    n_edges = torch.Tensor.size(edge_index)[1]
    
    # Generate random edge attributes
    edge_attr = torch.randn(n_edges)

    # Define graph with generated inputs
    graph = Data(x=x,
                 edge_index=edge_index,
                 edge_attr=edge_attr)
    # Moving data to device
    graph = graph.to(device)
    return graph


def diffuse_t_steps(batch_0, t_step, n_t_steps, alpha_decay, n_features=None):
    """Performs t forward steps of the diffusive Markov chain.
    
    G (t) = \\alpha (t) G (t-1) + \\sigma (t) N (t)
    
    with G a graph and N noise. If n_features defined, only the :n_features node features are diffused.

    Args:
        batch_0    (Batch): Batch of graphs to be diffused.
        alpha_t    (float): Constant which controls how much signal is retained.
        n_features (int):   Number of node features to be diffused (:n_features).
    Returns:
        graph_t (torch_geometric.data.Data): Diffused graph (step t).
    """

    # Clone the original batch of graphs to prevent in-place modifications
    batch_t = batch_0.clone()

    # Get alpha and sigma at t
    alpha_t = get_alpha_t(t_step, n_t_steps, alpha_decay)
    sigma_t = get_sigma_t(t_step, n_t_steps, alpha_decay)

    # Number of nodes and features per graph
    n_nodes    = batch_t.x.size(0)
    n_features = n_features if n_features is not None else batch_t.x.size(1)

    # Generate gaussian (normal) noise
    epsilon_t = get_random_graph(n_nodes, n_features, batch_t.edge_index)

    # Forward pass
    batch_t.x[:, :n_features] = alpha_t * batch_t.x[:, :n_features] + sigma_t * epsilon_t.x
    batch_t.edge_attr         = alpha_t * batch_t.edge_attr         + sigma_t * epsilon_t.edge_attr

    return batch_t, epsilon_t


def predict_noise(batch_t, node_model, edge_model):
    """Predicts noise given some batch of noisy graphs using specified models.

    Args:
        batch_t    (torch_geometric.data.Data): Batch with noisy undirected graphs, consistent with model definitions.
        node_model (torch.nn.Module):           Model for graph-node prediction.
        edge_model (torch.nn.Module):           Model for graph-edge prediction.
        

    Returns:
        pred_e_batch_t (torch_geometric.data.Data): Predicted noise for batch g_batch_t.
    """

    batch_0 = batch_t.clone()
    
    # Perform a single forward pass for predicting node features
    out_x = node_model(batch_t.x, batch_t.edge_index, batch_t.edge_attr)
    if torch.isnan(out_x).any():
        raise ValueError("Tensor out_x contains NaN values.")

    # Define x_i and x_j as features of every corresponding pair of nodes (same order than attributes)
    x_i = batch_t.x[batch_t.edge_index[0]]
    x_j = batch_t.x[batch_t.edge_index[1]]
    
    # Perform a single forward pass for predicting edge attributes
    # Introduce previous edge attributes as features as well
    out_attr = edge_model(x_i, x_j, batch_t.edge_attr).ravel()
    if torch.isnan(out_attr).any():
        raise ValueError("Tensor out_x contains NaN values.")

    # Update node features and edge attributes in g_batch_0 with the predicted out_x and out_attr
    batch_0.x         = out_x
    batch_0.edge_attr = out_attr
    return batch_0


def denoising_step(batch_t, epsilon_t, t_step, n_t_steps, alpha_decay, n_features=None):
    """Performs a forward step of a denoising chain.

    Args:
        batch_t    (Batch): Batch of graphs to be diffused.
        epsilon_t  (Batch): Predicted noise to subtract at step t.
        alpha_t    (float): Constant from the step of the diffusion process.
        sigma      (float): Parameter which controls the amount of noised added when generating.
        n_features (int):   Number of node features to be diffused (:n_features).

    Returns:
        graph_0 (torch_geometric.data.Data): Denoised graph (step t-1).
    """

    # Clone the original batch of graphs to prevent in-place modifications
    batch_s = batch_t.clone()

    # Number of nodes and features per graph
    n_nodes    = batch_s.x.size(0)
    n_features = n_features if n_features is not None else batch_s.x.size(1)
    
    # Generate gaussian (normal) noise
    epsilon = get_random_graph(n_nodes, n_features, batch_s.edge_index)

    alpha_t_s    = get_alpha_t_s(t_step, t_step-1, n_t_steps, alpha_decay) #
    sigma_t      = get_sigma_t(t_step, n_t_steps, alpha_decay)
    sigma_t_s    = get_sigma_t_s(t_step, t_step-1, n_t_steps, alpha_decay)
    sigma_t_to_s = get_sigma_t_to_s(t_step, t_step-1, n_t_steps, alpha_decay) #

    aux = sigma_t_s**2 / (alpha_t_s * sigma_t)
    
    sigma_s = get_sigma_t(t_step-1, n_t_steps, alpha_decay)
    alpha_t = get_alpha_t(t_step, n_t_steps, alpha_decay)
    alpha_s = get_alpha_t(t_step-1, n_t_steps, alpha_decay)
    #print('alpha_t', alpha_t, 'alpha_s', alpha_s, 'alpha_t_s', alpha_t_s, 'sigma_t', sigma_t, 'sigma_s', sigma_s, 'sigma_t_s', sigma_t_s, 'sigma_t_to_s', sigma_t_to_s, 'aux', aux)

    # Backward pass

    batch_s.x[:, :n_features] = batch_s.x[:, :n_features] / alpha_t_s - aux * epsilon_t.x         + sigma_t_to_s * epsilon.x
    batch_s.edge_attr         = batch_s.edge_attr         / alpha_t_s - aux * epsilon_t.edge_attr + sigma_t_to_s * epsilon.edge_attr

    
    return batch_s


def denoise(batch_t, n_t_steps, alpha_decay, node_model, edge_model, plot_steps=False, n_features=None):
    """Performs consecutive steps of diffusion in a reference batch of graphs.

    Args:
        batch_t     (Batch):           Reference batch of graphs to be denoised (step t-1).
        n_t_steps   (int):             Number of diffusive steps.
        alpha_decay (float):           Parameter which controls the decay of alpha with t.
        node_model  (torch.nn.Module): Model for graph-node prediction.
        edge_model  (torch.nn.Module): Model for graph-edge prediction.
        plot_steps  (bool, int):       Whether to plot each intermediate step, or which graph from batch.
        n_features  (int):             Number of node features to be diffused (:n_features).

    Returns:
        graph_0 (torch_geometric.data.Data): Graph with random node features and edge attributes (step t).
    """

    # Clone batch of graphs and move to device
    batch_s = batch_t.clone().to(device)

    for t_step in torch.arange(n_t_steps-1, -1, -1, device=device):
        # Standard normalization for the time step, which is added to node-level graph embeddings after
        t_step_std = t_step / n_t_steps - 0.5

        # Stack time step across batch dimension
        batch_s.x[:, -1] = t_step_std

        # Predict batch noise at given time step
        pred_epsilon_t = predict_noise(batch_s,
                                       node_model, edge_model)
        
        
        # Check if intermediate steps are plotted; then, plot the NetworkX graph
        if plot_steps:
            # Convert PyTorch graph to NetworkX graph
            networkx_graph = to_networkx(batch_s[plot_steps])
            pos            = nx.spring_layout(networkx_graph)
            nx.draw(networkx_graph, pos, with_labels=True, node_size=batch_s[plot_steps].x, font_size=10)
            plt.show()

        # Compute alpha_t and denoise batch altogether
        batch_s = denoising_step(batch_s, pred_epsilon_t,
                                 t_step, n_t_steps, alpha_decay,
                                 n_features=n_features)
       

    # Check if intermediate steps are plotted; then, plot the NetworkX graph
    if plot_steps:
        # Convert PyTorch graph to NetworkX graph
        networkx_graph = to_networkx(batch_s[plot_steps])
        pos            = nx.spring_layout(networkx_graph)
        nx.draw(networkx_graph, pos, with_labels=True, node_size=batch_s[plot_steps].x, font_size=10)
        plt.show()
    return batch_s
class nGCNN(torch.nn.Module):
    """Graph convolutional neural network for the prediction of node embeddings.
    The network consists of recursive convolutional layers, which input node features plus graph level embeddings
    while it outputs updated node level embeddings.
    """

    def __init__(self, n_node_features, n_graph_features, pdropout):
        super(nGCNN, self).__init__()

        # Set random seed for reproducibility
        torch.manual_seed(12345)

        # Define graph convolution layers
        # Introducing graph features
        self.conv1 = GraphConv(n_node_features+n_graph_features, 256)
        self.conv2 = GraphConv(256, 512)
        self.conv3 = GraphConv(512, 256)
        self.conv4 = GraphConv(256, n_node_features)  # Predict all node features at once

        # Normalization helps model stability
        self.norm1 = GraphNorm(256)

        self.pdropout = pdropout

    def forward(self, x, edge_index, edge_attr):
        # Apply graph convolution with ReLU activation function
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)
        if torch.isnan(x).any():
            raise ValueError(" In node it was nan before the norm. Tensor contains NaN values.")
        x = self.norm1(x)  # Batch normalization
        if torch.isnan(x).any():
            raise ValueError(" In node it was nan after the norm. Tensor contains NaN values.")
        x = x.relu()
        x = self.conv4(x, edge_index, edge_attr)
        return x
class eGCNN(nn.Module):
    """Convolutional neural network for the prediction of edge attributes.
    Predictions of the new link arise from the product of the two involved nodes and the previous edge attribute.
    The network consists of recursive convolutional layers, which input edge attribute plus graph level embeddings
    and plus previous edge attribute embeddings while it outputs updated attribute embeddings.
    """

    def __init__(self, n_node_features, n_graph_features, pdropout):
        super(eGCNN, self).__init__()

        # Set random seed for reproducibility
        torch.manual_seed(12345)

        # Define linear convolution layers
        # Introducing node features + previous edge attribute
        self.linear1 = Linear(n_node_features+n_graph_features+1, 128)
        self.linear2 = Linear(128, 256)
        self.linear3 = Linear(256, 64)
        self.linear4 = Linear(64, 1)  # Predicting one single weight

        # Normalization helps model stability
        self.norm1 = GraphNorm(64)
        
        self.pdropout = pdropout

    def forward(self, x_i, x_j, previous_attr):
        # Dot product between node distances
        x_i = torch.cat((torch.pow(x_i[:, :-1] - x_j[:, :-1], 2), x_i[:, -1:]), dim=1)  # Of dimension [..., features_channels]
        
        # Reshape previous_attr tensor to have the same number of dimensions as x
        previous_attr = previous_attr.view(-1, 1)  # Reshapes from [...] to [..., 1]

        # Concatenate the tensors along dimension 1 to get a tensor of size [..., num_embeddings ~ 6]
        x = torch.cat((x_i, previous_attr), dim=1)

        # Apply linear convolution with ReLU activation function
        x = self.linear1(x)

        # Dropout layer (only for training)
        x = F.dropout(x, p=self.pdropout, training=self.training)

        # Last linear convolution
        x = self.linear2(x)
        x = x.relu()
        x = self.linear3(x)
        if torch.isnan(x).any():
            raise ValueError(" In edge it was nan before the norm. Tensor contains NaN values.")
        x = self.norm1(x)  # Batch normalization
        if torch.isnan(x).any():
            raise ValueError(" In edge it was nan after the norm. Tensor contains NaN values.")
        x = x.relu()
        x = self.linear4(x)
        return x
class line_eGCNN(nn.Module):
    """Graph convolutional neural network for the prediction of edge attributes.
    In this implementation, we use line-graphs, exploiding the power of GNNs from link prediction.
    Predictions of the new link arise from the product of the two involved nodes and the previous edge attribute.
    The network consists of recursive convolutional layers, which input edge features plus graph level embeddings
    and plus previous edge attribute embeddings while it outputs updated attribute embeddings.
    """

    def __init__(self, n_node_features, n_graph_features, pdropout):
        super(line_eGCNN, self).__init__()

        # Set random seed for reproducibility
        torch.manual_seed(12345)

        self.linear1 = Linear(n_node_features + n_graph_features + 1,
                              128)  # Introducing node features + previous edge attribute
        self.linear2 = Linear(128, 64)  # Convolutional layer
        self.linear3 = Linear(64, 1)  # Predicting one single weight

        self.pdropout = pdropout

    def forward(self, x_i, x_j, previous_attr):
        # Dot product between node distances
        x_i[:, :-1] = torch.pow(x_i[:, :-1] - x_j[:, :-1], 2)  # Of dimension [..., features_channels]

        # Reshape previous_attr tensor to have the same number of dimensions as x
        previous_attr = previous_attr.view(-1, 1)  # Reshapes from [...] to [..., 1]

        # Concatenate the tensors along dimension 1 to get a tensor of size [..., num_embeddings ~ 6]
        x = torch.cat((x_i, previous_attr), dim=1)

        # Apply linear convolution with ReLU activation function
        x = self.linear1(x)

        # Dropout layer (only for training)
        x = F.dropout(x, p=self.pdropout, training=self.training)

        # Last linear convolution
        x = self.linear2(x)
        x = x.relu()
        x = self.linear3(x)
        return x


def get_graph_losses(graph1, graph2):
    """Calculate loss values for node features and edge attributes between two graphs.
    Depending on the size of the graphs, calculating MSE loss directly might be memory-intensive.
    Processing that in batches or subsets of nodes/edges can be more appropriate.

    Args:
        graph1 (torch_geometric.data.Data): The first input graph.
        graph2 (torch_geometric.data.Data): The second input graph.

    Returns:
        node_losses (list with torch.Tensor inside): Loss value for node features between the two graphs.
        edge_loss   (torch.Tensor):                         Loss value for edge attributes between the two graphs.
    """

    # Initialize loss criteria for nodes and edges
    node_criterion = nn.MSELoss()
    edge_criterion = nn.MSELoss()

    # Calculate the loss for node features by comparing the node attribute tensors
    node_losses = []
    for i in range(graph1.x.size(1)):
        node_loss = node_criterion(graph1.x[:, i],
                                   graph2.x[:, i])
        node_losses.append(node_loss)

    # Calculate the loss for edge attributes by comparing the edge attribute tensors
    edge_loss = edge_criterion(graph1.edge_attr,
                               graph2.edge_attr)

    return node_losses, edge_loss


def add_features_to_graph(graph_0, node_features):
    """Include some more information to the node features. The generated graph does not modify the input graph.

    Args:
        graph_0       (torch_geometric.data.Data): The input graph containing edge indexes and attributes.
        node_features (torch.array of size 1):     Information to be added to the graph (target,
                                                   step of the diffusing/denoising process, etc.).

    Returns:
        graph (torch_geometric.data.Data): Updated graph, with node_features as a new node feature for every atom.
    """

    graph = graph_0.clone()
    
    # Check that the size of node_features is the expected by the function
    if len(torch.Tensor.size(node_features)) != 1:
        sys.exit('Error: node_features does not have the expected size')
    
    # Concatenate tensors along the second dimension (dim=1) and update the graph with the new node features
    graph.x = torch.cat((graph.x, node_features.unsqueeze(0).repeat(graph.x.size(0), 1)), dim=1)
    return graph


def interpolate_graphs(dataset):
    """Linearly interpolates a set of graphs.

    Args:
        dataset (list): List containing graph structures.

    Returns:
        interp_dataset (list): List containing interpolated graph structures.
    """
    
    #graph_0 = zeros_like(dataset[0])
    
    return dataset
class EarlyStopping():
    def __init__(self, patience=5, delta=0, wandb_run=None, model_name='model.pt'):
        """Initializes the EarlyStopping object. Saves a model if accuracy is improved.
        Declares early_stop = True if training does not improve in patience steps within a delta threshold.

        Args:
            patience   (int):          Number of steps with no improvement.
            delta      (float):        Threshold for a score to be considered an improvement.
            wandb_run  (wandb object): Optional WandB run object for logging.
            model_name (str):          Name of the saved model checkpoint file.
        """
        self.patience = patience  # Number of steps with no improvement
        self.delta = delta  # Threshold for a score to be an improvement
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.wandb_run = wandb_run
        self.model_name = model_name

    def __call__(self, val_loss, model):
        """Call method to check and update early stopping.

        Args:
            val_loss (float):           Current validation loss.
            model    (torch.nn.Module): The PyTorch model being trained.
        """
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.wandb_run:
                self.wandb_run.log({'Early Stopping Counter': self.counter})

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Save the model checkpoint if the validation loss has decreased.
        It uses model.module, allowing models loaded to nn.DataParallel.

        Args:
            val_loss (float):           Current validation loss.
            model    (torch.nn.Module): The PyTorch model being trained.
        """
        if val_loss < self.val_loss_min:
            torch.save(model.module.state_dict(), self.model_name)
            self.val_loss_min = val_loss

class DenoisingModel():
    """
    Wrapper class for the denoising model, which includes the node and edge models, as well as the model configuration for training and inference

    Parameters
    ----------
    model_config : str
        Path to the YAML file with the model configuration.
    n_node_features : int
        Number of node features.
    n_graph_features : int #TODO: not sure what this is
        Number of graph features (outcome to be predicted).
    node_model : str, optional 
        Path to the node model checkpoint.
    edge_model : str, optional
        Path to the edge model checkpoint.
    device : str, optional
        Device to run the model on (default is 'cuda').

    """
    def __init__(self, model_config, n_node_features, n_graph_features, node_model_path=None, edge_model_path=None, device="cuda"):

        self.device = device

        self.n_node_features = n_node_features
        self.n_graph_features = n_graph_features

        # Load model configuration and set attributes
        with open(model_config, 'r') as file:
            config = yaml.safe_load(file)   

        # Dynamically create attributes from the YAML keys
        for key, value in config.items():
            # Convert specific keys to tensors if needed
            if key == "n_t_steps":
                value = torch.tensor(value, dtype=torch.int, device=self.device)
            elif key == "alpha_decay":
                value = torch.tensor(value, device=self.device)
            
            setattr(self, key, value)

        # Model definition
        # Instantiate the models for nodes and edges, considering the t_step information; n_graph_features+1 for accounting for the time step
        node_model = nGCNN(self.n_node_features, self.n_graph_features+1, self.dropout_node).to(self.device)
        edge_model = eGCNN(self.n_node_features, self.n_graph_features+1, self.dropout_edge).to(self.device)

        # Load previous model if available
        if node_model_path is not None and edge_model_path is not None:
            try:
                # Load and evaluate model state
                node_model.load_state_dict(torch.load(node_model_path, map_location=torch.device(self.device), weights_only=False))                
                edge_model.load_state_dict(torch.load(edge_model_path, map_location=torch.device(self.device), weights_only=False))
            except FileNotFoundError:
                pass

        # Allow data parallelization among multi-GPU
        node_model= nn.DataParallel(node_model)
        edge_model= nn.DataParallel(edge_model)

        self.node_model = node_model
        self.edge_model = edge_model

    def train(self, train_data, val_data, exp_name, val_jump=5, train_specific_step=None):
        """
        Train the denoising model using the provided data

        Parameters
        ----------
        train_data : torch_geometric.DataLoader
            DataLoader object with the training data
        val_data : torch_geometric.DataLoader
            DataLoader object with the validation data
        exp_name : str
            Name of the experiment directory
        val_jump : int
            Interval between epochs to perform validation
        train_specific_step : int, optional
            If provided, train the model only for the specific time step
        """
        # Create the experiment directory
        if not os.path.exists(exp_name):
            os.makedirs(exp_name)

        # Step 1: Set up the logger
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename=os.path.join(exp_name, "loss.log"), filemode='w')

        # Step 2: Create a logger object
        self.logger = logging.getLogger()

        # Initialize the optimizers
        node_optimizer = torch.optim.Adam(self.node_model.parameters(), lr=self.learning_rate)
        edge_optimizer = torch.optim.Adam(self.edge_model.parameters(), lr=self.learning_rate)
        # Initialize early stopping
        node_early_stopping = EarlyStopping(patience=self.patience, delta=self.delta, model_name=os.path.join(exp_name, 'node_model.pt'))
        edge_early_stopping = EarlyStopping(patience=self.patience, delta=self.delta, model_name=os.path.join(exp_name, 'edge_model.pt'))

        # Training loop
        node_train_losses = []
        edge_train_losses = []
        node_val_losses = []
        edge_val_losses = []
        node_val_losses_per_t_step = []
        edge_val_losses_per_t_step = []

        print("Starting training...")
        for epoch in range(self.n_epochs):

            # Set models to training mode
            self.node_model.train()
            self.edge_model.train()

            start = time.time()
            self.ground_truth = {k: [] for k in self.node_attributes}
            self.prediction   = {k: [] for k in self.node_attributes}

            node_train_loss, edge_train_loss = self.train_epoch(train_data, node_optimizer, edge_optimizer, node_early_stopping, edge_early_stopping, train_specific_step)
                        
            node_train_losses.append(node_train_loss)
            edge_train_losses.append(edge_train_loss)

            # Check early stopping criteria
            """
            node_early_stopping(node_train_loss.sum(), self.node_model)  #TODO: check if .sum() is required
            edge_early_stopping(edge_train_loss, self. edge_model)

            if node_early_stopping.early_stop and edge_early_stopping.early_stop:
                print('Early stopping')
                break
            """

            print_node_loss = ' '.join([f'{node_loss:.4f}' for node_loss in node_train_loss])
            print(f'Epoch: {epoch+1}, edge loss: {edge_train_loss:.4f}, node loss: {print_node_loss}. Time elapsed: {time.time()-start:.2f} seconds')
        """ #TODO: uncomment
            # Perform validation every N epoch
            if not (epoch+1) % val_jump:
                node_val_loss, edge_val_loss, node_val_loss_per_t_step, edge_val_loss_per_t_step = self.eval(val_data)

                node_val_losses.append(node_val_loss)
                edge_val_losses.append(edge_val_loss)
                node_val_losses_per_t_step.append(node_val_loss_per_t_step)
                edge_val_losses_per_t_step.append(edge_val_loss_per_t_step)

            print("---------------------------------------------------------")
        
        # Save losses to CSV files
        self.save_losses(node_train_losses, edge_train_losses, node_val_losses, edge_val_losses, node_val_losses_per_t_step, edge_val_losses_per_t_step, exp_name)

        # Plot losses
        self.plot_losses(node_train_losses, edge_train_losses, node_val_losses, edge_val_losses, exp_name)
        """

    def eval(self, val_data, time_step_jump=10):
        """
        Evaluate the model using the provided validation data

        Parameters
        ----------
        val_data : torch_geometric.DataLoader
            DataLoader object with the validation data
        time_step_jump : int
            Interval between time steps to evaluate the model

        Returns
        -------
        node_val_loss : float
            Node validation loss
        edge_val_loss : float
            Edge validation loss
        node_val_loss_per_t_step : dict
            Node validation loss per time step
        edge_val_loss_per_t_step : dict
            Edge validation loss per time step

        """
        # Initialize mean losses
        edge_test_losses = 0
        node_test_losses = np.zeros(self.n_node_features, dtype=float)

        # Initialize losses per time step
        node_test_losses_per_t_step = {}
        edge_test_losses_per_t_step = {}

        # Initialize the models in evaluation mode
        self.node_model.eval()
        self.edge_model.eval()
        start = time.time()
        # Perform validation
        explored_batches = 0
        with torch.no_grad():
            for batch_idx, batch_0 in enumerate(val_data):
                # Check loss at each time step
                explored_batches += 1
                n_steps_checked = 0
                for t_step in range(1,self.n_t_steps,time_step_jump): #TODO: change 2 to self.n_t_steps
                    g_batch_0 = batch_0.clone().to(self.device)
                    t_step_std =  t_step / self.n_t_steps - 0.5
                    g_batch_t, e_batch_t = diffuse_t_steps(g_batch_0, t_step, self.n_t_steps, self.alpha_decay, n_features=self.n_node_features)
                    g_batch_t.x[:, -1] = t_step_std
            
                    pred_epsilon_t = predict_noise(g_batch_t, self.node_model, self.edge_model)

                    # Calculate the losses for node features and edge attributes
                    node_losses, edge_loss = get_graph_losses(e_batch_t, pred_epsilon_t)  

                    # Add losses to the cumulative sum
                    node_test_losses += np.array([loss.item() for loss in node_losses])
                    edge_test_losses += edge_loss.item()

                    # Save losses per time step
                    
                    node_test_losses_per_t_step[t_step] = np.array([loss.item() for loss in node_losses]) / len(val_data)
                    edge_test_losses_per_t_step[t_step] = edge_loss.item() / len(val_data)

                    n_steps_checked += 1

        # Compute the average loss of all t's over the validation set
        # convert n_t_steps from tensor to int if needed
        if type(self.n_t_steps) == torch.Tensor:
            self.n_t_steps = self.n_t_steps.item()

        node_test_losses /=  (explored_batches * n_steps_checked)
        edge_test_losses /= (explored_batches * n_steps_checked)

        print_node_loss = ' '.join([f'{node_loss:.4f}' for node_loss in node_test_losses])
        print(f'Validation losses: edge loss: {edge_test_losses:.4f}, node loss: {print_node_loss}. Time elapsed: {time.time()-start:.2f} seconds')

        return node_test_losses, edge_test_losses, node_test_losses_per_t_step, edge_test_losses_per_t_step

    def train_epoch(self, train_data, node_optimizer, edge_optimizer, node_early_stopping, edge_early_stopping, train_specific_step=None):
        """Train the model for one epoch."""
        explored_batches = 0

        edge_loss_cum = 0
        node_loss_cum = np.zeros(self.n_node_features, dtype=float)

        for batch_idx, batch_0 in enumerate(train_data):
            if batch_idx == 1: #TODO: remove to use all data
                break
            g_batch_0 = batch_0.clone().to(self.device)
           
            # Repeat the batch 2^N times
            if self.n_repeat > 1:
                for _ in range(self.n_repeat):
                    # Repeat each element in g_batch_0 (this could be node features, edge indices, etc.)
                    g_batch_0.x = torch.cat([g_batch_0.x, g_batch_0.x], dim=0)  # Repeat node features
                    g_batch_0.edge_index = torch.cat([g_batch_0.edge_index, g_batch_0.edge_index], dim=1)  # Repeat edge indices
                    g_batch_0.edge_attr = torch.cat([g_batch_0.edge_attr, g_batch_0.edge_attr], dim=0) if g_batch_0.edge_attr is not None else None  # Repeat edge attributes
                    g_batch_0.batch = torch.cat([g_batch_0.batch, g_batch_0.batch + g_batch_0.x.size(0)], dim=0)  # Adjust the batch assignment

               
            explored_batches += 1 

            # Train for a specific time step if required
            if train_specific_step is not None:
                init_era = train_specific_step
                stop_era = train_specific_step + 1
            else:
                init_era = 0
                stop_era = self.n_eras

            for era in range(init_era, stop_era):
                # Randomly sample a time step and normalize it
                t_step = torch.randint(0, self.n_t_steps, (1,))[0]
                #t_step = era #TODO: remove
                t_step_std =  t_step / self.n_t_steps - 0.5

                node_optimizer.zero_grad()
                edge_optimizer.zero_grad()

                g_batch_t, e_batch_t = diffuse_t_steps(g_batch_0, t_step, self.n_t_steps, self.alpha_decay, n_features=self.n_node_features)
                g_batch_t.x[:, -1] = t_step_std #add time step to the last feature
            
                pred_epsilon_t = predict_noise(g_batch_t, self.node_model, self.edge_model)

                # Calculate the losses for node features and edge attributes
                node_losses, edge_loss = get_graph_losses(e_batch_t, pred_epsilon_t)
                
                # Combine losses for each attribute tensors
                node_loss = torch.stack(node_losses).sum()

                self.logger.info(f"Era: {era}, " + "edge_loss:" + f"{edge_loss.item()}" +  "," + "node_losses:" + ",".join(f"{loss.item()}" for loss in node_losses))  # TODO: remove if not required
                node_loss_cum += np.array([loss.item() for loss in node_losses])
                edge_loss_cum += edge_loss.item()
                
                self.optimize(node_loss, node_optimizer, self.node_model, max_norm=2.0, early_stopping=node_early_stopping)
                self.optimize(edge_loss, edge_optimizer, self.edge_model, max_norm=2.0, early_stopping=edge_early_stopping)
            print("-----------------------")

        node_loss_cum /= ((stop_era - init_era) * explored_batches)
        edge_loss_cum /= ((stop_era - init_era) * explored_batches)  
       
        return node_loss_cum, edge_loss_cum
    
    def compute_losses(self, e_batch_t, pred_epsilon_t):
        """Compute node and edge losses."""
        node_losses, edge_loss = get_graph_losses(e_batch_t, pred_epsilon_t)
        return torch.stack(node_losses).sum(), edge_loss

    def optimize(self, loss, optimizer, model, max_norm, early_stopping):
        """Optimize a given loss."""
        if not early_stopping.early_stop:
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()

    def save_losses(self, node_train_losses, edge_train_losses, node_val_losses, edge_val_losses, node_val_losses_per_t_step, edge_val_losses_per_t_step, exp_name):
        """
        Save losses to CSV files.

        Parameters
        ----------
        node_train_losses : list or array
            Node training losses.
        edge_train_losses : list or array
            Edge training losses.
        node_val_losses : list or array
            Node validation losses.
        edge_val_losses : list or array
            Edge validation losses.
        node_val_losses_per_t_step : list of dicts
            Node validation losses per time step. Each dictionary corresponds to an epoch contains the losses for each node attribute for all time steps.
        edge_val_losses_per_t_step : list of dicts
            Edge validation losses per time step. Each dictionary corresponds to an epoch contains the losses for each edge attribute for all time steps.
        """

        # Save mean losses
        np.savetxt(os.path.join(exp_name, 'node_train_losses.csv'), node_train_losses, delimiter=',')
        np.savetxt(os.path.join(exp_name, 'edge_train_losses.csv'), edge_train_losses, delimiter=',')
        np.savetxt(os.path.join(exp_name, 'node_val_losses.csv'), node_val_losses, delimiter=',')
        np.savetxt(os.path.join(exp_name, 'edge_val_losses.csv'), edge_val_losses, delimiter=',')

        # Save losses per time step
        data = []
        for epoch, losses_at_epoch in enumerate(node_val_losses_per_t_step):
            for t_step, losses in losses_at_epoch.items():
                # losses is assumed to be a list of node features
                data_entry = {'Epoch': epoch, 'Time Step': t_step}
                for feature_idx, loss in enumerate(losses):
                    data_entry[f'Feature_{feature_idx}'] = loss
                data.append(data_entry)

        node_val_losses_df = pd.DataFrame(data)
        node_val_losses_df.to_csv(os.path.join(exp_name, 'node_val_losses_per_t_step.csv'), index=False)

        data = []
        for epoch, loss_at_epoch in enumerate(edge_val_losses_per_t_step):
            for t_step, loss in loss_at_epoch.items():
                data_entry = {'Epoch': epoch, 'Time Step': t_step, 'Loss': loss}
            data.append(data_entry)

        edge_val_losses_df = pd.DataFrame(data)
        edge_val_losses_df.to_csv(os.path.join(exp_name, 'edge_val_losses_per_t_step.csv'), index=False)
                  
    def plot_losses(self, node_train_losses, edge_train_losses, node_val_losses, edge_val_losses, exp_name):
        """
        Plot and save the losses.

        Parameters
        ----------
        node_train_losses : list or array
            Node training losses.
        edge_train_losses : list or array
            Edge training losses.
        node_val_losses : list or array
            Node validation losses.
        edge_val_losses : list or array
            Edge validation losses.
        """
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        ax[0].plot(node_train_losses, label='Node train losses')
        ax[1].plot(node_val_losses, label='Node val losses')
        ax[2].plot(edge_train_losses, label='Edge train losses')
        ax[3].plot(edge_val_losses, label='Edge val losses')

        ax[0].set_title('Node train losses')
        ax[1].set_title('Node val losses')
        ax[2].set_title('Edge train losses')
        ax[3].set_title('Edge val losses')

        plt.tight_layout()
        plt.savefig(os.path.join(exp_name, 'losses.png'))
        plt.close(fig)

 
