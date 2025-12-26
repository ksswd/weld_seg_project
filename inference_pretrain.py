import os
import numpy as np
import torch
from model.model import GeometryAwareTransformer
from train.mask_strategy import HighCurvatureMasker
from utils.config import Config

def save_point_cloud(file_path, point_cloud):
    """Save point cloud to a .npz file."""
    np.savez(file_path, features=point_cloud)

def load_model(config, weights_path):
    """Load the pretrained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeometryAwareTransformer(config).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def infer_and_compare(file_path, model, masker, output_dir):
    """Perform inference and compare masked and reconstructed point clouds."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the input point cloud
    data = np.load(file_path)
    features = torch.tensor(data['features'], dtype=torch.float32).to(device)
    curvature = torch.tensor(data['curvature'], dtype=torch.float32).to(device)
    print(curvature.shape)
    # Add batch dimension (B=1) to curvature
    curvature = curvature.unsqueeze(0)

    # Ensure curvature has the correct shape (B, N, 1) before passing to the model
    curvature = curvature.unsqueeze(-1) if curvature.ndim == 2 else curvature

    # Debugging: Print curvature shape before passing to the model
    print(f"Curvature shape before model: {curvature.shape}")

    # Generate mask
    mask = masker.generate_mask(curvature).squeeze(0).squeeze(-1).bool()

    # Mask the input point cloud
    masked_features = features.clone()
    masked_features[..., :3][mask] = 0.0

    # Save the masked point cloud
    masked_output_path = os.path.join(output_dir, "masked_point_cloud.npz")
    save_point_cloud(masked_output_path, masked_features.cpu().numpy())

    # Perform inference
    with torch.no_grad():
        reconstructed_features = model(masked_features.unsqueeze(0),
                                        principal_dir=torch.tensor(data['principal_dir'], dtype=torch.float32).unsqueeze(0).to(device),
                                        curvature=torch.tensor(data['curvature'], dtype=torch.float32).unsqueeze(0).to(device),
                                        density=torch.tensor(data['local_density'], dtype=torch.float32).unsqueeze(0).to(device),
                                        normals=torch.tensor(data['normals'], dtype=torch.float32).unsqueeze(0).to(device),
                                        linearity=torch.tensor(data['linearity'], dtype=torch.float32).unsqueeze(0).to(device),
                                        task='recon').squeeze(0)

    # Save the reconstructed point cloud
    reconstructed_output_path = os.path.join(output_dir, "reconstructed_point_cloud.npz")
    save_point_cloud(reconstructed_output_path, reconstructed_features.cpu().numpy())

    # Compare masked and reconstructed point clouds
    masked_coords = features[mask, :3].cpu().numpy()
    reconstructed_coords = reconstructed_features[mask, :3].cpu().numpy()
    coord_diff = np.linalg.norm(masked_coords - reconstructed_coords, axis=1)

    # Save the coordinate differences
    diff_output_path = os.path.join(output_dir, "coordinate_differences.npy")
    np.save(diff_output_path, coord_diff)

    print(f"Masked point cloud saved to: {masked_output_path}")
    print(f"Reconstructed point cloud saved to: {reconstructed_output_path}")
    print(f"Coordinate differences saved to: {diff_output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inference script for self-supervised pretraining validation.")
    parser.add_argument("--file", type=str, required=True, help="Path to the input .npz file.")
    parser.add_argument("--weights", type=str, required=True, help="Path to the pretrained model weights.")
    parser.add_argument("--output", type=str, required=True, help="Directory to save the outputs.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load configuration and model
    config = Config
    model = load_model(config, args.weights)

    # Initialize masker
    masker = HighCurvatureMasker(mask_ratio=config.MASK_RATIO)

    # Perform inference and comparison
    infer_and_compare(args.file, model, masker, args.output)