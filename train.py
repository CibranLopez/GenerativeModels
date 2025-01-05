"""
python train.py --model-config model_configs/config.yaml --data-path data/Loaded_MP_bandgap-sphere-images_standardized --dest-path exps/train/prueba1 --shuffle --pin-memory --check-labels 
"""

import argparse
from libraries.model import DenoisingModel
from libraries.dataloader import MPStandardizedDataloader

import time

def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Train a denoising model with specified configurations.")
    
    # Add arguments
    parser.add_argument("--model-config", type=str, help="Path to the model configuration YAML file.")
    parser.add_argument("--node-model-path", type=str, default=None, help="Path to the node model. Default is None.")
    parser.add_argument("--edge-model-path", type=str, default=None, help="Path to the edge model. Default is None.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the dataset for training and validation.")
    parser.add_argument("--dest-path", type=str, required=True, help="Destination path to save the training outputs.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of workers for the dataloader.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for the dataloader.")
    parser.add_argument("--shuffle", action="store_true", help="Whether to shuffle the dataset.")
    parser.add_argument("--pin-memory", action="store_true", help="Whether to pin memory for the dataloader.")
    parser.add_argument("--train-ratio", type=float, default=0.98, help="Ratio of the dataset to use for training.")
    parser.add_argument("--check-labels", action="store_true", help="Whether to use predefined labels to split data instead of train ratio.")
    parser.add_argument("--train-portion", type=float, default=0.1, help="Portion of subset to be used for training.")
    parser.add_argument("--valid-portion", type=float, default=1, help="Portion of subset to be used for validation.")
    parser.add_argument("--test-portion", type=float, default=1, help="Portion of subset to be used for testing.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    print("Arguments:", args)
    # Load the data 
    dataloader = MPStandardizedDataloader(
        data_path=args.data_path, 
        batch_size=args.batch_size, 
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    
    train_dataloader, val_dataloader, test_dataloader = dataloader.get_dataloaders(train_ratio=args.train_ratio, check_labels=args.check_labels, train_portion=args.train_portion, valid_portion=args.valid_portion, test_portion=args.test_portion)

    # Initialize the model
    model = DenoisingModel(
        node_model_path=args.node_model_path, 
        edge_model_path=args.edge_model_path, 
        model_config=args.model_config,
        n_node_features=4, #TODO: fix hardcode
        n_graph_features=1) #TODO fix hardcode
    
    # Train the model
    start = time.time()
    model.train(train_dataloader, val_dataloader, exp_name=args.dest_path, val_jump=10)
    end = time.time()
    print("Training time:", end - start)

if __name__ == "__main__":
    main()
