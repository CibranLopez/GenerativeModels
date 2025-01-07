import numpy               as np
import matplotlib.pyplot   as plt
import torch.nn.functional as F
import torch.nn            as nn
import networkx            as nx
import torch
import sys

from torch_geometric.data          import Data, Batch
from torch.nn                      import Linear
from torch_geometric.nn            import GraphConv
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

    # Generate random node features
    x = torch.randn(n_nodes, n_features)

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


def predict_noise(batch_t, model):
    """Predicts noise given some batch of noisy graphs using specified models.

    Args:
        batch_t (torch_geometric.data.Data): Batch with noisy undirected graphs, consistent with model definitions.
        model   (torch.nn.Module):           Model for graph noise prediction.

    Returns:
        pred_e_batch_t (torch_geometric.data.Data): Predicted noise for batch g_batch_t.
    """

    # Perform a single forward pass for predicting node features
    return model(batch_t)


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

    alpha_t_s    = get_alpha_t_s(t_step, t_step-1, n_t_steps, alpha_decay)
    sigma_t      = get_sigma_t(t_step, n_t_steps, alpha_decay)
    sigma_t_s    = get_sigma_t_s(t_step, t_step-1, n_t_steps, alpha_decay)
    sigma_t_to_s = get_sigma_t_to_s(t_step, t_step-1, n_t_steps, alpha_decay)

    aux = sigma_t_s**2 / (alpha_t_s * sigma_t)

    # Backward pass
    batch_s.x[:, :n_features] = batch_s.x[:, :n_features] / alpha_t_s - aux * epsilon_t.x         + sigma_t_to_s * epsilon.x
    batch_s.edge_attr         = batch_s.edge_attr         / alpha_t_s - aux * epsilon_t.edge_attr + sigma_t_to_s * epsilon.edge_attr
    return batch_s


def denoise(batch_t, n_t_steps, alpha_decay, model, plot_steps=False, n_features=None):
    """Performs consecutive steps of diffusion in a reference batch of graphs.

    Args:
        batch_t     (Batch):           Reference batch of graphs to be denoised (step t-1).
        n_t_steps   (int):             Number of diffusive steps.
        alpha_decay (float):           Parameter which controls the decay of alpha with t.
        model       (torch.nn.Module): Model for graph-noise prediction.
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
        pred_epsilon_t = predict_noise(batch_s, model)
        
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
        #print(batch_s.x[:2])
        
    # Check if intermediate steps are plotted; then, plot the NetworkX graph
    if plot_steps:
        # Convert PyTorch graph to NetworkX graph
        networkx_graph = to_networkx(batch_s[plot_steps])
        pos            = nx.spring_layout(networkx_graph)
        nx.draw(networkx_graph, pos, with_labels=True, node_size=batch_s[plot_steps].x, font_size=10)
        plt.show()
    return batch_s


class GNN(torch.nn.Module):
    """
    Combined Graph Convolutional Neural Network for node and edge prediction.
    Alternately updates node and edge embeddings after each convolutional layer.
    """

    def __init__(self, n_node_features, n_graph_features, pdropout_node, pdropout_edge):
        super(GNN, self).__init__()

        torch.manual_seed(12345)

        neurons_n_1 = 128
        neurons_n_2 = 256
        neurons_n_3 = 64

        neurons_e_1 = 64
        neurons_e_2 = 128
        neurons_e_3 = 64
        neurons_e_4 = 32

        # Node update layers (GraphConv)
        self.node_conv1 = GraphConv(n_node_features + n_graph_features, neurons_n_1)
        self.node_conv2 = GraphConv(neurons_n_1, neurons_n_2)
        self.node_conv3 = GraphConv(neurons_n_2, neurons_n_3)
        self.node_conv4 = GraphConv(neurons_n_3, n_node_features)

        # Edge update layers (Linear)
        self.edge_linear_f1 = Linear(n_node_features+n_graph_features+1, neurons_e_1)  # From ini to multi
        self.edge_linear_r1 = Linear(neurons_e_1, 1)  # From multi to 1

        self.edge_linear_f2 = Linear(neurons_n_1+1, neurons_e_2)  # From ini to multi
        self.edge_linear_r2 = Linear(neurons_e_2, 1)  # From multi to 1

        self.edge_linear_f3 = Linear(neurons_n_2+1, neurons_e_3)  # From ini to multi
        self.edge_linear_r3 = Linear(neurons_e_3, 1)  # From multi to 1

        self.edge_linear_f4 = Linear(neurons_n_3+1, neurons_e_4)  # From ini to multi
        self.edge_linear_r4 = Linear(neurons_e_4, 1)  # From multi to 1
        
        # Normalization layers
        self.node_norm1 = torch.nn.BatchNorm1d(256)
        self.edge_norm1 = torch.nn.BatchNorm1d(64)

        self.pdropout_node = pdropout_node
        self.pdropout_edge = pdropout_edge

    def forward(self, batch):
        """
        Perform forward propagation alternately updating nodes and edges.

        Args:
            batch: A batch object containing x, edge_index, and edge_attr.
            graph_features: Graph-level features tensor.

        Returns:
            Updated batch object with updated x and edge_attr.
        """

        # Update 1
        x         = self.node_forward(batch, self.node_conv1)
        edge_attr = self.edge_forward(batch, self.edge_linear_f1, self.edge_linear_r1)
        batch.x, batch.edge_attr = x, edge_attr

        # Update 2
        x         = self.node_forward(batch, self.node_conv2)
        edge_attr = self.edge_forward(batch, self.edge_linear_f2, self.edge_linear_r2)
        batch.x, batch.edge_attr = x, edge_attr

        # Update 3
        x         = self.node_forward(batch, self.node_conv3)
        edge_attr = self.edge_forward(batch, self.edge_linear_f3, self.edge_linear_r3)
        batch.x, batch.edge_attr = x, edge_attr

        # Update 4
        x         = self.node_forward(batch, self.node_conv4, activation_function=False)
        edge_attr = self.edge_forward(batch, self.edge_linear_f4, self.edge_linear_r4)
        batch.x, batch.edge_attr = x, edge_attr
        return batch

    def node_forward(self, batch, node_conv, activation_function=True):
        """
        Update node embeddings using the current node features and edge attributes.

        Args:
            batch: Batch object containing x, edge_index, and edge_attr.
            node_conv: Graph convolutional layer.

        Returns:
            Updated node embeddings.
        """
        # Read properties from the batch object
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

        x = node_conv(x, edge_index, edge_attr)
        if activation_function:
            x = x.relu()
        return x

    def edge_forward(self, batch, edge_linear_forward, edge_linear_reverse):
        """
        Update edge attributes using the current node features and edge attributes.

        Args:
            batch: Batch object containing x, edge_index, and edge_attr.
            edge_linear: Linear layer for edge attribute prediction in multi-dimensional space.
            edge_linear_reverse: Move back to 1-dimensional edge attributes.

        Returns:
            Updated edge attributes.
        """
        # Read properties from the batch object
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

        # Define x_i and x_j as features of every corresponding pair of nodes (same order than attributes)
        x_i = x[edge_index[0]]
        x_j = x[edge_index[1]]

        # Reshape previous_attr tensor to have the same number of dimensions as x
        previous_attr = edge_attr.view(-1, 1)  # Reshapes from [...] to [..., 1]

        # Calculate squared distance between node features
        x_diff = torch.pow(x_i[:, :-1] - x_j[:, :-1], 2)

        # Concatenate node differences, edge attributes, and graph features
        edge_attr = torch.cat((x_diff, x_i[:, -1:]),
                              dim=1)  # Of dimension [..., features_channels]

        # Concatenate the tensors along dimension 1 to get a tensor of size [..., num_embeddings ~ 6]
        edge_attr = torch.cat((edge_attr, previous_attr), dim=1)

        # Apply linear convolution with ReLU activation function
        edge_attr = edge_linear_forward(edge_attr)
        edge_attr = edge_attr.relu()
        edge_attr = edge_linear_reverse(edge_attr).ravel()
        return edge_attr


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
        self.val_loss_min = np.Inf
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
