import json
import numpy as np
import os 
from torch_geometric.loader import DataLoader
import torch

from libraries.dataset import standardize_dataset, get_datasets
from libraries.model import add_features_to_graph

class MPStandardizedDataloader():
    """
    Dataloader for the MPStandardizedDataset.

    It assumes that the dataset is already standardized and that the following files are present in the data_path:
    - standardized_dataset.pt
    - standardized_labels.pt
    - standardized_parameters.json
    - train_labels.txt
    - validation_labels.txt
    - test_labels.txt

    Parameters
    ----------
    data_path : str
        Path to the directory containing the dataset files.
    batch_size : int
        Batch size for the dataloader.
    shuffle : bool
        Whether to shuffle the dataset.
    num_workers : int
        Number of workers for the dataloader (   # Set num_workers > 0 if multithread is possible, otherwise set to 0)
    pin_memory : bool
        Whether to pin memory for the dataloader.
    """

    def __init__(self, data_path, batch_size, shuffle=True, num_workers=0, pin_memory=True):
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.dataset_name = os.path.join(self.data_path, "standardized_dataset.pt")
        self.dataset_parameters_name = os.path.join(self.data_path, "standardized_parameters.json")
        self.labels_name = os.path.join(self.data_path, "standardized_labels.pt")

        with open(self.dataset_parameters_name, 'r') as json_file:
            numpy_dict = json.load(json_file)
        
        # Convert NumPy arrays back to PyTorch tensors
        self.dataset_parameters = {}
        for key, value in numpy_dict.items():
            try:
                self.dataset_parameters[key] = torch.tensor(value)
            except:
                self.dataset_parameters[key] = value

    def get_dataloaders(self, train_ratio=0.8, check_labels=False):

        print("Loading dataset...")
        dataset = torch.load(self.dataset_name, weights_only=False)
        
        # Make room for n_graph_features and t_steps in the dataset
        for idx in range(len(dataset)):
            dataset[idx] = add_features_to_graph(dataset[idx],
                                                torch.tensor([dataset[idx].y, 0]))


        print("Generating dataloaders...")
        if check_labels:     
            labels  = torch.load(self.labels_name,  weights_only=False)

            # Load the labels
            path_to_train_labels = os.path.join(self.data_path, 'train_labels.txt')
            path_to_val_labels   = os.path.join(self.data_path, 'validation_labels.txt')
            path_to_test_labels  = os.path.join(self.data_path, 'test_labels.txt')

            train_labels = np.genfromtxt(path_to_train_labels, dtype='str').tolist()
            val_labels   = np.genfromtxt(path_to_val_labels,   dtype='str').tolist()
            test_labels  = np.genfromtxt(path_to_test_labels,  dtype='str').tolist()

            # Use the computed indexes to generate train and test sets
            # We iteratively check where labels equals a unique train/test labels and append the index to a list
            material_labels = labels.copy()
            train_dataset = get_datasets(train_labels, material_labels, dataset)
            val_dataset = get_datasets(val_labels,  material_labels, dataset)
            test_dataset  = get_datasets(test_labels,  material_labels, dataset)
        else:
            # Define the sizes of the train and test sets
            train_size = int(train_ratio * len(dataset))
            test_size  = int((1- train_ratio) / 2  * len(dataset)) # remaining of the split is divided into 2 equal parts for validation and test

            # Shuffle the dataset
            np.random.seed(42)
            np.random.shuffle(dataset)
            train_dataset = dataset[:train_size]
            val_dataset = dataset[train_size:-test_size]
            test_dataset = dataset[-test_size:]

        # Create the dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader   = DataLoader(val_dataset,   batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        test_loader  = DataLoader(test_dataset,  batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)

        return train_loader, val_loader, test_loader
