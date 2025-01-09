import torch
import torch.nn as nn
from torch_geometric.nn import GraphConv, GraphNorm
from torch_geometric.data import Data
from torchinfo import summary
import torch 
from libraries.model import nGCNN


data = torch.load("data/Loaded_MP_bandgap-voronoi/standardized_dataset.pt")
data_sample = data[0]


n_node_features = 4  # 4 node features
n_graph_features = 2  # 2 graph features
pdropout = 0.5  # Dropout rate

# Instantiate the model
model = nGCNN(n_node_features, n_graph_features, pdropout)


# Pack the data into a Data object for PyG usage


# Display the model summary
summary(model, input_data=(data_sample.x, data_sample.edge_index, data_sample.edge_attr, data_sample.batch), depth=2)
