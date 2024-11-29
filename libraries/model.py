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


def get_alpha_t(t, T, s):
    """Defines constant alpha at time-step t, given a parameter s < 0.5 (else alpha increases).
    
    \\alpha (t) = (1 - 2 s) \\left( 1 - \\left( \\frac{t}{T} \\right)^2 \\right) + s

    Args:
        t (int):   time step (of diffusion or denoising) in which alpha is required.
        T (int):   total number of steps.
        s (float): parameter which controls the decay of alpha with t.

    Returns:
        alpha (float): parameter which controls the velocity of diffusion or denoising.
    """

    #return torch.tensor((1 - 2 * s) * (1 - (t / T) ** 2) + 2 * s)
    return (1 - 2 * s) * (1 - (t / (T+1))**2) + s


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


def diffusion_step(batch_0, alpha_t, n_features=None):
    """Performs a forward step of a diffusive, Markov chain.
    
    G (t) = \\sqrt{\\alpha (t)} G (t-1) + \\sqrt{1 - \\alpha (t)} N (t)
    
    with G a graph and N noise. If n_features defined, only the :n_features node features are diffused.

    Args:
        batch_0    (Batch): Batch of graphs to be diffused.
        alpha_t    (float): Constant from the step of the diffusion process.
        n_features (int):   Number of node features to be diffused (:n_features).
    Returns:
        graph_t (torch_geometric.data.Data): Diffused graph (step t).
    """

    # Clone the original batch of graphs to prevent in-place modifications
    batch_t = batch_0.clone()

    # Number of nodes and features per graph
    n_nodes    = batch_t.x.size(0)
    n_features = n_features if n_features is not None else batch_t.x.size(1)

    # Generate gaussian (normal) noise
    epsilon_t = get_random_graph(n_nodes, n_features, batch_t.edge_index)

    # Forward pass
    batch_t.x[:, :n_features] = torch.sqrt(alpha_t) * batch_t.x[:, :n_features] + torch.sqrt(1-alpha_t) * epsilon_t.x
    batch_t.edge_attr         = torch.sqrt(alpha_t) * batch_t.edge_attr         + torch.sqrt(1-alpha_t) * epsilon_t.edge_attr
    return batch_t, epsilon_t


def diffuse(batch_0, n_t_steps, alpha_decay=1e-2, plot_steps=False, ouput_all_graphs=False, n_features=None):
    """Performs consecutive steps of diffusion in a reference batch of graphs.

    Args:
        batch_0     (Batch): Reference batch of graphs to be diffused (step t-1).
        n_t_steps   (int):   Number of diffusive steps.
        alpha_decay (float): Parameter which controls the decay of alpha with t.
        plot_steps  (bool):  Whether to plot or not each intermediate step.
        n_features  (int):   Number of node features to be diffused (:n_features).

    Returns:
        graph_t (torch_geometric.data.Data): Graph with random node features and edge attributes (step t).
    """

    # Clone batch of graphs
    batch_t = batch_0.clone()

    # Append all graphs for debugging
    if ouput_all_graphs:
        all_graphs = []
    
    # Define t_steps starting from 1 to n_t_steps+1
    for t_step in torch.arange(n_t_steps, device=device):
        # Check if intermediate steps are plotted; then, plot the NetworkX graph
        if plot_steps:
            # Convert PyTorch graph to NetworkX graph
            networkx_graph = to_networkx(batch_t)
            pos            = nx.spring_layout(networkx_graph)
            nx.draw(networkx_graph, pos, with_labels=True, node_size=batch_t.x.size()[1], font_size=10)
            plt.show()

        # Compute alpha_t and diffuse batch altogether
        alpha_t = get_alpha_t(t_step, n_t_steps, alpha_decay)
        batch_t, _ = diffusion_step(batch_t, alpha_t, n_features)
        
        if ouput_all_graphs:
            all_graphs.append(batch_t)
    
    # Check if intermediate steps are plotted; then, plot the NetworkX graph
    if plot_steps:
        # Convert PyTorch graph to NetworkX graph
        networkx_graph = to_networkx(batch_t)
        pos            = nx.spring_layout(networkx_graph)
        nx.draw(networkx_graph, pos, with_labels=True, node_size=batch_t.x, font_size=10)
        plt.show()
    if ouput_all_graphs:
        return batch_t, all_graphs
    return batch_t


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

    # Define x_i and x_j as features of every corresponding pair of nodes (same order than attributes)
    x_i = batch_t.x[batch_t.edge_index[0]]
    x_j = batch_t.x[batch_t.edge_index[1]]
    
    # Perform a single forward pass for predicting edge attributes
    # Introduce previous edge attributes as features as well
    out_attr = edge_model(x_i, x_j, batch_t.edge_attr).ravel()

    # Update node features and edge attributes in g_batch_0 with the predicted out_x and out_attr
    batch_0.x         = out_x
    batch_0.edge_attr = out_attr
    return batch_0


def denoising_step(batch_t, epsilon, alpha_t, sigma, n_features=None):
    """Performs a forward step of a denoising chain.

    Args:
        batch_t    (Batch): Batch of graphs to be diffused.
        epsilon    (Batch): Predicted noise to subtract.
        alpha_t    (float): Constant from the step of the diffusion process.
        sigma      (float): Parameter which controls the amount of noised added when generating.
        n_features (int):   Number of node features to be diffused (:n_features).

    Returns:
        graph_0 (torch_geometric.data.Data): Denoised graph (step t-1).
    """

    # Clone the original batch of graphs to prevent in-place modifications
    batch_0 = batch_t.clone()

    # Number of nodes and features per graph
    n_nodes    = batch_t.x.size(0)
    n_features = n_features if n_features is not None else batch_t.x.size(1)
    
    # Generate gaussian (normal) noise
    epsilon_t = get_random_graph(n_nodes, n_features, batch_0.edge_index)
    
    # Backward pass
    batch_0.x[:, :n_features] = batch_0.x[:, :n_features] / torch.sqrt(alpha_t) - torch.sqrt((1 - alpha_t) / alpha_t) * epsilon.x         + sigma * epsilon_t.x
    batch_0.edge_attr         = batch_0.edge_attr         / torch.sqrt(alpha_t) - torch.sqrt((1 - alpha_t) / alpha_t) * epsilon.edge_attr + sigma * epsilon_t.edge_attr
    return batch_0


def denoise(batch_t, n_t_steps, node_model, edge_model, alpha_decay=1e-2, sigma=None, plot_steps=False, n_features=None):
    """Performs consecutive steps of diffusion in a reference batch of graphs.

    Args:
        batch_t     (Batch):           Reference batch of graphs to be denoised (step t-1).
        n_t_steps   (int):             Number of diffusive steps.
        node_model  (torch.nn.Module): Model for graph-node prediction.
        edge_model  (torch.nn.Module): Model for graph-edge prediction.
        alpha_decay (float):           Parameter which controls the decay of alpha with t.
        sigma       (float):           Parameter which controls the amount of noised added when generating.
        plot_steps  (bool, int):       Whether to plot each intermediate step, or which graph from batch.
        n_features  (int):             Number of node features to be diffused (:n_features).

    Returns:
        graph_0 (torch_geometric.data.Data): Graph with random node features and edge attributes (step t).
    """

    # Clone batch of graphs and move to device
    batch_0 = batch_t.clone().to(device)

    for t_step in torch.arange(n_t_steps, device=device):
        # Standard normalization for the time step, which is added to node-level graph embeddings after
        t_step_std = t_step / n_t_steps - 0.5

        # Stack time step across batch dimension
        batch_0.x[:, -1] = t_step_std

        # Predict batch noise at given time step
        pred_epsilon_t = predict_noise(batch_0, node_model, edge_model)
        
        # Check if intermediate steps are plotted; then, plot the NetworkX graph
        if plot_steps:
            # Convert PyTorch graph to NetworkX graph
            networkx_graph = to_networkx(batch_0[plot_steps])
            pos            = nx.spring_layout(networkx_graph)
            nx.draw(networkx_graph, pos, with_labels=True, node_size=batch_0[plot_steps].x, font_size=10)
            plt.show()

        # Compute alpha_t and denoise batch altogether
        alpha_t = get_alpha_t(t_step, n_t_steps, alpha_decay)
        batch_0 = denoising_step(batch_0, pred_epsilon_t, alpha_t, sigma, n_features=n_features)
        
        print()
        print('Step: ', t_step)
        print('Alpha: ', alpha_t, 1/torch.sqrt(alpha_t), torch.sqrt((1 - alpha_t) / alpha_t))
        print(get_random_graph(batch_0.x.size(0), n_features if n_features is not None else batch_0.x.size(1), batch_0.edge_index).x[:5])
        print('Pred epsilon: ', pred_epsilon_t.x[:5])
        print('Resulting batch: ', batch_0.x[:5])
        
    # Check if intermediate steps are plotted; then, plot the NetworkX graph
    if plot_steps:
        # Convert PyTorch graph to NetworkX graph
        networkx_graph = to_networkx(batch_0[plot_steps])
        pos            = nx.spring_layout(networkx_graph)
        nx.draw(networkx_graph, pos, with_labels=True, node_size=batch_0[plot_steps].x, font_size=10)
        plt.show()
    return batch_0


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
        self.norm1 = torch.nn.BatchNorm1d(256)

        self.pdropout = pdropout

    def forward(self, x, edge_index, edge_attr):
        # Apply graph convolution with ReLU activation function
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)
        x = self.norm1(x)  # Batch normalization
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
        self.norm1 = torch.nn.BatchNorm1d(64)
        
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
        x = self.norm1(x)  # Batch normalization
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
