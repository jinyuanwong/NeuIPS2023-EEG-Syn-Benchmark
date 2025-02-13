import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf
from generative.networks.nets import AutoencoderKL
from dataset.dataset_fnirs import create_dataloaders
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="./config/config_aekl_eeg_fnirs.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint"
    )
    parser.add_argument(
        "--path_train_data",
        type=str,
        default='path/to/train_data.npy',
        help="Path to fNIRS training data"
    )
    parser.add_argument(
        "--path_train_labels",
        type=str,
        default='path/to/train_labels.npy',
        help="Path to fNIRS training labels"
    )
    parser.add_argument(
        "--path_valid_data",
        type=str,
        default='path/to/valid_data.npy',
        help="Path to fNIRS validation data"
    )
    parser.add_argument(
        "--path_valid_labels",
        type=str,
        default='path/to/valid_labels.npy',
        help="Path to fNIRS validation labels"
    )
    return parser.parse_args()

def visualize_reconstruction(model, data_batch, save_dir):
    """Visualize original and reconstructed signals"""
    model.eval()
    with torch.no_grad():
        # Move data to same device as model
        data = data_batch['input_data'].to(next(model.parameters()).device)
        
        # Get reconstruction and latent representations
        reconstruction, z_mu, z_sigma = model(data)
        
        # Move tensors to CPU for visualization
        data = data.cpu().numpy()
        reconstruction = reconstruction.cpu().numpy()
        z_mu = z_mu.cpu().numpy()
        
        # Print shapes
        print(f"\n=== Data Shapes ===")
        print(f"Original data shape: {data.shape}")
        print(f"Reconstructed data shape: {reconstruction.shape}")
        print(f"Latent representation shape: {z_mu.shape}")
        
        # Plot first sample's channels
        n_channels = min(4, data.shape[1])  # Plot first 4 channels or less
        fig, axes = plt.subplots(n_channels, 1, figsize=(15, 3*n_channels))
        if n_channels == 1:
            axes = [axes]
            
        for i in range(n_channels):
            axes[i].plot(data[0, i, :], label='Original', alpha=0.7)
            axes[i].plot(reconstruction[0, i, :], label='Reconstructed', alpha=0.7)
            axes[i].set_title(f'Channel {i+1}')
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'reconstruction_comparison.png')
        plt.close()
        
        # Visualize latent space
        plt.figure(figsize=(10, 4))
        plt.imshow(z_mu[0], aspect='auto', interpolation='nearest')
        plt.colorbar(label='Latent Value')
        plt.title('Latent Space Representation (First Sample)')
        plt.xlabel('Latent Dimension')
        plt.ylabel('Channel')
        plt.savefig(save_dir / 'latent_space.png')
        plt.close()

def main():
    args = parse_args()
    
    # Create save directory with proper path handling
    save_dir = Path("./outputs/fnirs_visualisation/")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = OmegaConf.load(args.config_file)
    
    # Initialize model
    model = AutoencoderKL(**config.autoencoderkl.params)
    
    # Load trained weights with error handling
    try:
        checkpoint = torch.load(args.model_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Successfully loaded model weights")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Load data with validation set
    data_paths = {
        'train_data': args.path_train_data,
        'train_labels': args.path_train_labels,
        'valid_data': args.path_valid_data,
        'valid_labels': args.path_valid_labels
    }
    
    try:
        _, valid_loader = create_dataloaders(
            data_paths,
            batch_size=1,
            num_workers=1
        )
        first_batch = next(iter(valid_loader))
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # Visualize reconstruction
    visualize_reconstruction(model, first_batch, save_dir)
    
    print(f"\nVisualization results saved in: {save_dir.absolute()}")

if __name__ == "__main__":
    main()
