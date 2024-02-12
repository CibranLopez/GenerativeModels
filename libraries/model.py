import numpy               as np
import matplotlib.pyplot   as plt
import torch.nn.functional as F
import torch.nn            as nn
import networkx            as nx
import torch
import sys

from torch_geometric.data          import Data
from torch.nn                      import Linear
from torch_geometric.nn            import GraphConv
from torch_geometric.utils.convert import to_networkx

# Checking if pytorch can run in GPU, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    graph = Data(x=x,
                 edge_index=edge_index,
                 edge_attr=edge_attr)
    return graph


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
    
    # Define t_steps starting from 1 to n_t_steps+1
    t_steps = np.arange(1, n_diffusing_steps+1)
    for t in t_steps:
        # Check if intermediate steps are plotted; then, plot the NetworkX graph
        if plot_steps:
            # Convert PyTorch graph to NetworkX graph
            networkx_graph = to_networkx(graph_t)
            pos            = nx.spring_layout(networkx_graph)
            nx.draw(networkx_graph, pos, with_labels=True, node_size=graph_t.x, font_size=10)
            plt.show()
        
        graph_t, _ = diffusion_step(graph_t, t, n_diffusing_steps, s)
        all_graphs.append(graph_t)
    
    # Check if intermediate steps are plotted; then, plot the NetworkX graph
    if plot_steps:
        # Convert PyTorch graph to NetworkX graph
        networkx_graph = to_networkx(graph_t)
        pos            = nx.spring_layout(networkx_graph)
        nx.draw(networkx_graph, pos, with_labels=True, node_size=graph_t.x, font_size=10)
        plt.show()
    return graph_t, all_graphs


def diffusion_step(graph_0, t, n_diffusing_steps, s):
    """Performs a forward step of a diffusive, Markov chain.
    
    G (t) = \sqrt{\alpha (t)} G (t-1) + \sqrt{1 - \alpha (t)} N (t)
    
    with G a graph and N noise.

    Args:
        graph_0           (torch_geometric.data.Data): Graph which is to be diffused (step t-1).
        t                 (float):                     Step of the diffusion process.
        n_diffusing_steps (int):                       Number of diffusion steps.
        s                 (float):                     Parameter which controls the decay of alpha with t.
    Returns:
        graph_t (torch_geometric.data.Data): Diffused graph (step t).
    """

    # Clone graph that we are diffusing (not strictly necessary)
    graph_t = graph_0.clone()

    # Number of nodes and features in the graph
    n_nodes, n_features = torch.Tensor.size(graph_t.x)

    # Generate gaussian (normal) noise
    epsilon_t = get_random_graph(n_nodes, n_features, graph_t.edge_index)

    # Compute alpha_t
    alpha_t = get_alpha_t(t, n_diffusing_steps, s)

    # Forward pass
    graph_t.x         = torch.sqrt(alpha_t) * graph_t.x         + torch.sqrt(1 - alpha_t) * epsilon_t.x
    graph_t.edge_attr = torch.sqrt(alpha_t) * graph_t.edge_attr + torch.sqrt(1 - alpha_t) * epsilon_t.edge_attr
    return graph_t, epsilon_t


def denoise(graph_t, n_t_steps, node_model, edge_model, s=1e-2, sigma=None, plot_steps=False, target=None):
    """Performs consecutive steps of diffusion in a reference graph.

    Args:
        graph_t    (torch_geometric.data.Data): Reference graph to be diffused (step t-1).
        n_t_steps  (int):                       Number of diffusive steps.
        s          (float):                     Parameter which controls the decay of alpha with t.
        sigma      (float):                     Parameter which controls the amount of noised added when generating.
        plot_steps (bool):                      Whether to plot or not each intermediate step.
        target     (torch.array):               Information of the seeked target, to be added to the graph.

    Returns:
        graph_0 (torch_geometric.data.Data): Graph with random node features and edge attributes (step t).
    """
    
    graph_0 = graph_t.clone()
    
    # Save all intermediate graphs as well (sanity check)
    all_graphs = [graph_0]
    
    # Define t_steps as inverse of the diffuse process
    t_steps = np.arange(1, n_t_steps+1)[::-1]
    for t_step in t_steps:
        # Add t_step information to graph_t
        t_step_std = (t_step/n_t_steps - 0.5)  # Standard normalization
        graph_0 = add_features_to_graph(graph_0,
                                        torch.tensor([[t_step_std]], dtype=torch.float))
        
        # Add target information
        if target is not None:
            graph_0 = add_features_to_graph(graph_0,
                                            target)

        # Perform a single forward pass for predicting node features
        out_x = node_model(graph_0.x,
                           graph_0.edge_index,
                           graph_0.edge_attr)

        # Remove t_step information
        out_x = out_x[:, :-1]

        # Define x_i and x_j as features of every corresponding pair of nodes (same order than attributes)
        x_i = graph_0.x[graph_0.edge_index[0]]
        x_j = graph_0.x[graph_0.edge_index[1]]

        # Perform a single forward pass for predicting edge attributes
        # Introduce previous edge attributes as features as well
        out_attr = edge_model(x_i, x_j, graph_t.edge_attr)

        # Construct noise graph
        noise_graph = Data(x=out_x,
                           edge_index=graph_0.edge_index,
                           edge_attr=out_attr.ravel())
        
        # Check if intermediate steps are plotted; then, plot the NetworkX graph
        if plot_steps:
            # Convert PyTorch graph to NetworkX graph
            networkx_graph = to_networkx(graph_0)
            pos            = nx.spring_layout(networkx_graph)
            nx.draw(networkx_graph, pos, with_labels=True, node_size=graph_0.x, font_size=10)
            plt.show()

        # Remove t_step information from graph_0
        graph_0.x = graph_0.x[:, :-1]

        # Denoise the graph with the predicted noise
        graph_0 = denoising_step(graph_0, noise_graph, t_step, n_t_steps, s=s, sigma=sigma)
        all_graphs.append(graph_0)
        
    # Check if intermediate steps are plotted; then, plot the NetworkX graph
    if plot_steps:
        # Convert PyTorch graph to NetworkX graph
        networkx_graph = to_networkx(graph_0)
        pos            = nx.spring_layout(networkx_graph)
        nx.draw(networkx_graph, pos, with_labels=True, node_size=graph_0.x, font_size=10)
        plt.show()
    return graph_0, all_graphs


def denoising_step(graph_t, epsilon, t, n_t_steps, s, sigma):
    """Performs a forward step of a denoising chain.

    Args:
        graph_t  (torch_geometric.data.Data): Graph which is to be denoised (step t).
        epsilon  (torch_geometric.data.Data): Predicted noise to subtract.
        t        (int):                       Step of the diffusion process.
        n_t_steps (int):                      Number of diffusive steps.
        s        (float):                     Parameter which controls the decay of alpha with t.
        sigma    (float):                     Parameter which controls the amount of noised added when generating.

    Returns:
        graph_0 (torch_geometric.data.Data): Denoised graph (step t-1).
    """

    # Clone graph that we are denoising (not strictly necessary)
    graph_0 = graph_t.clone()

    # Compute alpha_t
    alpha_t = get_alpha_t(t, n_t_steps, s)

    # Number of nodes and features in the graph
    n_nodes, n_features = torch.Tensor.size(graph_t.x)
    
    # Generate gaussian (normal) noise
    epsilon_t = get_random_graph(n_nodes, n_features, graph_t.edge_index)
    
    # Backard pass
    graph_0.x         = graph_0.x         / torch.sqrt(alpha_t) - torch.sqrt((1 - alpha_t) / alpha_t) * epsilon.x         + sigma * epsilon_t.x
    graph_0.edge_attr = graph_0.edge_attr / torch.sqrt(alpha_t) - torch.sqrt((1 - alpha_t) / alpha_t) * epsilon.edge_attr + sigma * epsilon_t.edge_attr
    return graph_0


class nGCNN(torch.nn.Module):
    """Graph convolution neural network for the prediction of node embeddings.
    The network consists of recursive convolutional layers, which input node features plus graph level embeddings
    while it outputs updated node level embeddings.
    """

    def __init__(self, n_node_features, n_graph_features, pdropout):
        super(nGCNN, self).__init__()

        # Set random seed for reproducibility
        torch.manual_seed(12345)

        # Define graph convolution layers
        self.conv1 = GraphConv(n_node_features+n_graph_features, 256)  # Introducing node features
        self.conv2 = GraphConv(256, n_node_features)  # Predicting node features

        self.pdropout = pdropout

    def forward(self, x, edge_index, edge_attr):
        # Apply graph convolution with ReLU activation function
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        return x


class eGCNN(nn.Module):
    """Graph convolution neural network for the prediction of edge attributes.
    Predictions of the new link arise from the product of the two involved nodes and the previous edge attribute.
    The network consists of recursive convolutional layers, which input node features plus graph level embeddings
    and plus previous edge attribute embeddings while it outputs updated attribute embeddings.
    """

    def __init__(self, n_node_features, n_graph_features, pdropout):
        super(eGCNN, self).__init__()

        # Set random seed for reproducibility
        torch.manual_seed(12345)

        self.linear1 = Linear(n_node_features+n_graph_features+1, 64)  # Introducing node features + previous edge attribute
        self.linear2 = Linear(64, 1)  # Predicting one single weight

        self.pdropout = pdropout

    def forward(self, x_i, x_j, previous_attr):
        # Dot product between node distances (?)
        x_ij = x_i * x_j  # Of dimension [..., features_channels]

        # Reshape previous_attr tensor to have the same number of dimensions as x
        previous_attr = previous_attr.view(-1, 1)  # Reshapes from [...] to [..., 1]

        # Concatenate the tensors along dimension 1 to get a tensor of size [..., num_embeddings ~ 6]
        x = torch.cat((x_ij, previous_attr), dim=1)

        # Apply linear convolution with ReLU activation function
        x = self.linear1(x)

        # Dropout layer (only for training)
        x = F.dropout(x, p=self.pdropout, training=self.training)

        # Last linear convolution
        x = self.linear2(x)
        x = x.relu()
        return x


def get_graph_losses(graph1, graph2, batch_size):
    """Calculate loss values for node features and edge attributes between two graphs.
    Depending on the size of the graphs, calculating MSE loss directly might be memory-intensive.
    Processing that in batches or subsets of nodes/edges can be more appropriate.

    Args:
        graph1     (torch_geometric.data.Data): The first input graph.
        graph2     (torch_geometric.data.Data): The second input graph.
        batch_size (int):                       Size of the data batch, used to compute the MSE loss.

    Returns:
        node_loss (torch.Tensor): Loss value for node features between the two graphs.
        edge_loss (torch.Tensor): Loss value for edge attributes between the two graphs.
    """

    # Initialize loss criteria for nodes and edges
    node_criterion = nn.MSELoss()
    edge_criterion = nn.MSELoss()

    # Calculate the loss for node features by comparing the node attribute tensors
    node_loss = node_criterion(graph1.x,
                               graph2.x)

    # Calculate the loss for edge attributes by comparing the edge attribute tensors
    edge_loss = edge_criterion(graph1.edge_attr,
                               graph2.edge_attr)

    # Divide by the number of data graphs in the batch
    node_loss /= batch_size
    edge_loss /= batch_size

    return node_loss, edge_loss


def get_target_loss(obtained_target, seeked_target):
    """Calculate the target loss based on obtained and seeked targets.
    It checks if seeked_target is a specific value or rather a limit

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


def add_features_to_graph(graph_0, node_features):
    """Include some more information to the node features. The generated graph does not modify the input graph.

    Args:
        graph_0       (torch_geometric.data.Data): The input graph containing edge indexes and attributes.
        node_features (torch.array of size 2):     Information to be added to the graph (target,
                                                   step of the diffusing/denoising process, etc.).

    Returns:
        graph (torch_geometric.data.Data): Updated graph, with node_features as a new node feature for every atom.
    """

    graph = graph_0.clone()
    
    # Check that the size of node_features is the expected by the function
    if len(torch.Tensor.size(node_features)) != 1:
        sys.exit('Error: node_features does not have the expected size')
    
    # Concatenate tensors along the second dimension (dim=1)
    new_x = torch.cat((graph.x, node_features.unsqueeze(0).repeat(graph.x.size(0), 1)), dim=1)

    # Update the graph with the new node features
    graph.x = new_x
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
