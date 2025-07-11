import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
import glob
import random
from typing import List, Tuple
import math
import tqdm
from torch.amp import autocast, GradScaler
from scipy.signal import butter, filtfilt

# --- Configuration ---
# Directory where the processed .h5 files from your script are stored.
PROCESSED_DATA_DIR = 'data/cdl-projects/test1-haag-2025-05-21/processed'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Path to save the best performing encoder weights
BEST_ENCODER_SAVE_PATH = 'best_pose_encoder.pth'
# Path to save the final encoder weights after all epochs
FINAL_ENCODER_SAVE_PATH = 'final_pose_encoder.pth'

VELOCITY_KEY = 'velocity_filtered_butter_x_y_conf'
POSITION_KEY = 'position_filtered_x_y_conf'
VIDEO_FPS = 120
MASK_VELOCITY_FILTER_CUTOFF = 30
MASK_AVE_VELOCITY_THRESHOLD = 0.5
MASK_POSITION_AVE_CONFIDENCE_THRESHOLD = 0.5


# --- Data & Masking Parameters ---
WINDOW_SIZE = 90      # Number of time steps (frames) in each data instance.
STRIDE = 1            # Step size to move the window across the time series.
NUM_LANDMARKS = 26    # Number of landmarks in the data.
NUM_CHANNELS = 6     # x, y, confidence
MASK_RATIO = 0.5     # Fraction of landmarks to mask in each frame.
SCALE_RANGE = None #(0.75, 1.25)
ROTATION_RANGE = None #(-180, 180) #(-15, 15)

# --- Training Parameters ---
TEST_SPLIT = 0.2      # 20% of the files will be used for the test set.
BATCH_SIZE = 256
NUM_EPOCHS = 40
LEARNING_RATE = 1e-4
LR_EPOCH_MILESTONES = [25, 35]
# Set NUM_WORKERS to 0 if you are on Windows or debugging.
NUM_WORKERS = 10 if DEVICE == "cuda" else 0 

# --- Model Hyperparameters ---
POSE_EMBEDDING_DIM = 256
POSE_NUM_HEADS = 8
POSE_NUM_LAYERS = 3
POSE_DIM_FEEDFORWARD = 512

TIME_EMBEDDING_DIM = 256
TIME_NUM_HEADS = 8
TIME_NUM_LAYERS = 3
TIME_DIM_FEEDFORWARD = 512



class VelocityWindowDataset(Dataset):
    """
    PyTorch Dataset for loading windowed velocity data.

    For each sample, it randomly masks a specified ratio of landmarks at
    each timestep and returns only the visible (unmasked) data along with
    their original indices. It uses 'tube masking' where the same landmarks
    are masked across all timesteps in the window.

    Args:
        file_paths (List[str]): Paths to the HDF5 files.
        window_size (int): The number of time steps in each sample.
        stride (int): The step size for creating windows.
        num_landmarks (int): Total number of landmarks available.
        mask_ratio (float): Fraction of landmarks to mask at each timestep.
    """
    def __init__(self, 
                 file_paths: List[str], 
                 window_size: int, stride: int, 
                 num_landmarks: int, mask_ratio: float, 
                 scale_range: Tuple[float, float]=None,
                 rotation_range: Tuple[float, float]=None,
                ):
        super().__init__()
        self.file_paths = file_paths
        self.window_size = window_size
        self.stride = stride
        self.num_landmarks = num_landmarks
        self.mask_ratio = mask_ratio
        self.scale_range = scale_range
        self.num_masked = int(self.num_landmarks * self.mask_ratio)
        self.rotation_range = rotation_range
        
        self.indices = []
        
        print(f"Initializing dataset with {len(self.file_paths)} files...")
        self._create_indices()

    def _apply_filter(self, data):
        """Applies a low-pass filter to the velocity data."""
        nyquist = 0.5 * VIDEO_FPS
        normal_cutoff = MASK_VELOCITY_FILTER_CUTOFF / nyquist
        b, a = butter(2, normal_cutoff, btype='low', analog=False)
        T, L, C = data.shape
        flat = data.reshape(T, -1)          # (T, L*C)
        filtered = filtfilt(b, a, flat, axis=0)
        return filtered.reshape(T, L, C)

    def _create_indices(self):
        """Pre-calculates indices for every possible window for fast access."""
        for i, file_path in enumerate(self.file_paths):
            try:
                with h5py.File(file_path, 'r') as f:
                    velocity_data = f[VELOCITY_KEY][:]
                    # velocity_data[:, :, :2] = self._apply_filter(velocity_data[:, :, :2])
                    position_data = f[POSITION_KEY][:]
                    # mask for valid "center" frames is position confidence > 0.5 and speed > 0.5
                    valid_center_mask = (position_data[:, :, 2].mean(axis=1) > MASK_POSITION_AVE_CONFIDENCE_THRESHOLD) & (np.linalg.norm(velocity_data[:, :, :2], axis=2).mean(axis=1) > MASK_AVE_VELOCITY_THRESHOLD)
                    n_frames = velocity_data.shape[0]
                    n_valid_frames = np.sum(valid_center_mask)
                    if n_frames < self.window_size:
                        continue
                    if n_valid_frames < 1:
                        continue
                    center_frame_indices = np.where(valid_center_mask)[0]
                    start_indices = center_frame_indices - self.window_size // 2
                    start_indices = np.clip(start_indices, 0, n_frames - self.window_size)
                    start_indices = np.unique(start_indices)
                    for start_idx in start_indices:
                        self.indices.append((i, start_idx))
            except Exception as e:
                print(f"Warning: Could not process file {file_path}. Error: {e}")

    def __len__(self) -> int:
        """Returns the total number of windows (samples) in the dataset."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves one window of data, applies a random tube mask, and returns
        the visible data, their original indices, the original unmasked data,
        and the boolean mask.
        """
        file_idx, start_frame = self.indices[idx]
        file_path = self.file_paths[file_idx]
        
        with h5py.File(file_path, 'r') as f:
            end_frame = start_frame + self.window_size
            velocity_data = f[VELOCITY_KEY][start_frame:end_frame, :, :]
            # concatenate position and velocity
            position_data = f[POSITION_KEY][start_frame:end_frame, :, :]
            center_frame_pos = position_data[self.window_size // 2, :, :2]
            position_data[:, :, :2] -= center_frame_pos
            window_data = np.concatenate([position_data, velocity_data], axis=2)
            # channels:
            # 0,1: position x,y
            # 2: position confidence
            # 3,4: velocity x,y
            # 5: velocity confidence

        
        true_data = torch.from_numpy(window_data).float()
        if self.scale_range is not None:
            min_scale, max_scale = self.scale_range
            scale_factor = torch.rand(1) * (max_scale - min_scale) + min_scale
            true_data[..., :2] *= scale_factor
            true_data[..., 3:5] *= scale_factor

        if self.rotation_range is not None:
            min_rotation, max_rotation = self.rotation_range
            rotation_angle = torch.rand(1) * (max_rotation - min_rotation) + min_rotation
            rotation_rad = torch.deg2rad(rotation_angle)
            rotation_matrix = torch.tensor([
                [torch.cos(rotation_rad), -torch.sin(rotation_rad)],
                [torch.sin(rotation_rad), torch.cos(rotation_rad)]
            ])
            true_data[..., :2] = torch.matmul(true_data[..., :2], rotation_matrix)
            true_data[..., 3:5] = torch.matmul(true_data[..., 3:5], rotation_matrix)

        
        pos = true_data[..., :2]
        pos_mean, pos_std = pos.mean(), pos.std().clamp_min(1e-6)
        true_data[..., :2] = (pos - pos_mean) / pos_std
        vel = true_data[..., 3:5]
        vel_mean, vel_std = vel.mean(), vel.std().clamp_min(1e-6)
        true_data[..., 3:5] = (vel - vel_mean) / vel_std

        # Create a boolean mask of shape (T, L)
        # True means the landmark is masked (hidden)
        mask = torch.zeros(self.window_size, self.num_landmarks, dtype=torch.bool)
        
        # --- UPDATED: Tube Masking ---
        # Choose a single set of landmarks to mask across the entire time window.
        masked_landmark_indices = torch.randperm(self.num_landmarks)[:self.num_masked]
        # Apply this mask to all time steps.
        mask[:, masked_landmark_indices] = True
            
        # --- Prepare visible tokens and their indices ---
        visible_mask = ~mask  # Invert mask to get visible tokens

        # Get the actual data for visible tokens
        # This will have shape (num_visible_tokens, C)
        visible_data = true_data[visible_mask]

        # Get the original flat indices of the visible tokens
        T, L = self.window_size, self.num_landmarks
        flat_indices = torch.arange(T * L).view(T, L)
        visible_indices = flat_indices[visible_mask]

        # Returns:
        # 1. visible_data: The unmasked landmark data (N_visible, C)
        # 2. visible_indices: The original flat indices of the visible landmarks (N_visible,)
        # 3. true_data: The original complete data for loss calculation (T, L, C)
        # 4. mask: The boolean mask for loss calculation (T, L)
        return visible_data, visible_indices, true_data, mask

class PoseEncoder(nn.Module):
    def __init__(self,
                 window_size:int, num_landmarks:int, num_channels:int,
                 pose_embedding_dim:int, pose_num_heads:int,
                 pose_num_layers:int, pose_dim_feedforward:int,
                 temporal_stage_embedding_dim:int, time_num_heads:int,
                 time_num_layers:int, time_dim_feedforward:int):
        super().__init__()
        self.window_size = window_size
        self.num_landmarks = num_landmarks
        self.num_channels  = num_channels

        # ───── Landmark-stage ─────────────────────────────────────────────
        self.landmark_embedding = nn.Parameter(
            torch.randn(num_landmarks, pose_embedding_dim))
        self.pose_query = nn.Parameter(
            torch.randn(window_size, pose_embedding_dim))
        self.input_projection = nn.Linear(num_channels, pose_embedding_dim)

        self.landmark_to_pose_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(pose_embedding_dim,
                                       pose_num_heads,
                                       pose_dim_feedforward,
                                       dropout=0.1,
                                       activation='relu',
                                       batch_first=True),
            pose_num_layers)

        # ───── Temporal-stage positional encoding ────────────────────────
        position = torch.arange(window_size).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, temporal_stage_embedding_dim, 2) *
            (-math.log(10000.0) / temporal_stage_embedding_dim))
        pe = torch.zeros(window_size, temporal_stage_embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("time_positional_embedding", pe, persistent=False)

        self.pose1_to_pose2_projection = (nn.Identity() if
                                          pose_embedding_dim == temporal_stage_embedding_dim
                                          else nn.Linear(pose_embedding_dim,
                                                         temporal_stage_embedding_dim))

        self.pose_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(temporal_stage_embedding_dim,
                                       time_num_heads,
                                       time_dim_feedforward,
                                       dropout=0.1,
                                       activation='relu',
                                       batch_first=True),
            time_num_layers)

    # ──────────────────────────────────────────────────────────────────────
    def forward(self,
                visible_data  : torch.Tensor,   # (B, N_vis, C)
                visible_indices: torch.Tensor) -> torch.Tensor:
        B, N_vis, C = visible_data.shape

        # Project raw inputs and add learnable landmark embeddings
        x = self.input_projection(visible_data)
        landmark_idx = visible_indices % self.num_landmarks
        x = x + self.landmark_embedding[landmark_idx]

        # ***NEW: add temporal positional embedding per visible token***
        time_idx = visible_indices // self.num_landmarks   # 0‥T-1
        x = x + self.time_positional_embedding[time_idx]

        # Landmark-to-pose decoding
        tgt = self.pose_query.unsqueeze(0).expand(B, -1, -1)
        pose_tokens = self.landmark_to_pose_decoder(tgt=tgt, memory=x)

        # Temporal encoder
        pose_tokens = self.pose1_to_pose2_projection(pose_tokens)
        pose_tokens = pose_tokens + self.time_positional_embedding.unsqueeze(0)
        pose_tokens = self.pose_encoder(pose_tokens)
        return pose_tokens

class PoseMAEModel(torch.nn.Module):
    def __init__(self, 
                 window_size: int, 
                 num_landmarks: int, 
                 num_channels: int,
                 pose_embedding_dim: int,
                 pose_num_heads: int,
                 pose_num_layers: int,
                 pose_dim_feedforward: int,
                 temporal_stage_embedding_dim: int,
                 time_num_heads: int,
                 time_num_layers: int,
                 time_dim_feedforward: int,
                 
                 ):
        super().__init__()
        self.window_size = window_size
        self.num_landmarks = num_landmarks
        self.num_channels = num_channels

        self.pose_encoder = PoseEncoder(
            window_size,
            num_landmarks,
            num_channels,
            pose_embedding_dim,
            pose_num_heads,
            pose_num_layers,
            pose_dim_feedforward,
            temporal_stage_embedding_dim,
            time_num_heads,
            time_num_layers,
            time_dim_feedforward,
        )

        self.pose_output_decoder = nn.Sequential(
            nn.Linear(temporal_stage_embedding_dim, 2*num_landmarks*num_channels),
            nn.ReLU(),
            nn.Linear(2*num_landmarks*num_channels, num_landmarks*num_channels)
        )

    def forward(self, visible_data: torch.Tensor, visible_indices: torch.Tensor) -> torch.Tensor:
        """
        Processes a sequence of only visible tokens.
        
        Args:
            visible_data (torch.Tensor): Batch of visible landmark data.
                                         Shape: (B, N_visible, C).
            visible_indices (torch.Tensor): Original flat indices of visible landmarks.
                                            Shape: (B, N_visible).
        """
        pose_tokens = self.pose_encoder(visible_data, visible_indices)
        
        # 6. Decode pose tokens to reconstruct the full landmark sequence
        out = self.pose_output_decoder(pose_tokens).view(visible_data.size(0), self.window_size, self.num_landmarks, self.num_channels)
        return out


def confidence_weighted_masked_rmse_loss(
        predicted: torch.Tensor, 
        true: torch.Tensor, 
        mask: torch.Tensor) -> torch.Tensor:
    """
    Calculates RMSE loss only on the masked elements, weighted by confidence.
    
    Args:
        predicted (torch.Tensor): Model output, shape (B, T, L, C).
        true (torch.Tensor): Ground truth data, shape (B, T, L, C).
        mask (torch.Tensor): Boolean mask, True for masked elements, shape (B, T, L).

    Returns:
        torch.Tensor: A scalar loss value.
    """
    # Isolate the velocity components (x, y) and the confidence
    pred_vel = predicted[..., 3:5]
    true_vel = true[..., 3:5]
    true_conf = true[..., 5] # Shape: (B, T, L)
    pred_conf = predicted[..., 5]
    true_pos = true[..., :2]
    pred_pos = predicted[..., :2]

    # Calculate squared error for velocity components
    error_vel = (pred_vel - true_vel)**2
    error_pos = (pred_pos - true_pos)**2
    
    # Sum the error across the channels (x, y, x, y)
    vel_error_per_landmark = error_vel.sum(dim=-1) # Shape: (B, T, L)
    pos_error_per_landmark = error_pos.sum(dim=-1) # Shape: (B, T, L)

    # Weight the error by the confidence score
    vel_weighted_error = vel_error_per_landmark * true_conf
    pos_weighted_error = pos_error_per_landmark * true_conf
    
    # Only consider the error for the landmarks that were masked
    vel_masked_error = vel_weighted_error[mask]
    vel_inverse_masked_error = vel_weighted_error[~mask]
    vel_rmse = torch.sqrt(vel_masked_error.sum() / (mask.sum() + 1e-8))
    vel_inverse_rmse = torch.sqrt(vel_inverse_masked_error.sum() / (~mask).sum() + 1e-8)

    pos_masked_error = pos_weighted_error[mask]
    pos_inverse_masked_error = pos_weighted_error[~mask]
    pos_rmse = torch.sqrt(pos_masked_error.sum() / (mask.sum() + 1e-8))
    pos_inverse_rmse = torch.sqrt(pos_inverse_masked_error.sum() / (~mask).sum() + 1e-8)

    conf_loss = torch.nn.functional.mse_loss(pred_conf, true_conf, reduction='mean')
    
    rmse = (vel_rmse + pos_rmse) / 2
    inverse_rmse = (vel_inverse_rmse + pos_inverse_rmse) / 2

    return rmse + 0.005 * conf_loss + 0.05 * inverse_rmse
        
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # 1. Find all processed HDF5 files
    if not os.path.isdir(PROCESSED_DATA_DIR):
        raise FileNotFoundError(f"Directory not found: '{PROCESSED_DATA_DIR}'")
        
    all_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, '*.h5'))
    if not all_files:
        raise FileNotFoundError(f"No '.h5' files found in '{PROCESSED_DATA_DIR}'.")

    # 2. Create the train/test split at the FILE level
    random.seed(42)
    random.shuffle(all_files)
    all_files = all_files#[:100]
    
    split_index = int(len(all_files) * (1 - TEST_SPLIT))
    train_files = all_files[:split_index]
    test_files = all_files[split_index:]

    print(f"\nTotal files found: {len(all_files)}")
    print(f"Training files: {len(train_files)}")
    print(f"Testing files: {len(test_files)}")
    print("-" * 30)

    # 3. Create the Dataset objects
    train_dataset = VelocityWindowDataset(
        file_paths=train_files, window_size=WINDOW_SIZE, stride=STRIDE,
        num_landmarks=NUM_LANDMARKS, mask_ratio=MASK_RATIO, scale_range=SCALE_RANGE,
        rotation_range=ROTATION_RANGE
    )
    
    test_dataset = VelocityWindowDataset(
        file_paths=test_files, window_size=WINDOW_SIZE, stride=STRIDE,
        num_landmarks=NUM_LANDMARKS, mask_ratio=MASK_RATIO
    )
    
    print(f"\nTotal training samples (windows): {len(train_dataset)}")
    print(f"Total testing samples (windows): {len(test_dataset)}")
    print("-" * 30)

    # 4. Create the DataLoader objects
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # 5. Initialize Model, Optimizer, and Loss
    model = PoseMAEModel(
        window_size=WINDOW_SIZE,
        num_landmarks=NUM_LANDMARKS,
        num_channels=NUM_CHANNELS,
        pose_embedding_dim=POSE_EMBEDDING_DIM,
        pose_num_heads=POSE_NUM_HEADS,
        pose_num_layers=POSE_NUM_LAYERS,
        pose_dim_feedforward=POSE_DIM_FEEDFORWARD,
        temporal_stage_embedding_dim=TIME_EMBEDDING_DIM,
        time_num_heads=TIME_NUM_HEADS,
        time_num_layers=TIME_NUM_LAYERS,
        time_dim_feedforward=TIME_DIM_FEEDFORWARD
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_EPOCH_MILESTONES, gamma=0.2)
    scaler = GradScaler()
    loss_fn = confidence_weighted_masked_rmse_loss
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")
    print("-" * 30)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf') # Initialize best validation loss

    # 6. Training and Validation Loop
    for epoch in range(NUM_EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]", leave=False)
        
        for visible_data, visible_indices, true_data, mask in train_pbar:
            visible_data = visible_data.to(DEVICE)
            visible_indices = visible_indices.to(DEVICE)
            true_data = true_data.to(DEVICE)
            mask = mask.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type='cuda'):
                # Forward pass with the new data format
                predicted_data = model(visible_data, visible_indices)
                
                # Calculate loss using the original true data and mask
                loss = loss_fn(predicted_data, true_data, mask)
            
            # Backward pass and optimization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm.tqdm(test_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]", leave=False)

        with torch.no_grad():
            for visible_data, visible_indices, true_data, mask in val_pbar:
                visible_data = visible_data.to(DEVICE)
                visible_indices = visible_indices.to(DEVICE)
                true_data = true_data.to(DEVICE)
                mask = mask.to(DEVICE)
                
                with autocast(device_type='cuda'):
                    predicted_data = model(visible_data, visible_indices)
                    loss = loss_fn(predicted_data, true_data, mask)
                
                val_loss += loss.item()
                val_pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_val_loss = val_loss / len(test_loader)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Avg Train Loss: {avg_train_loss:.4f} | "
              f"Avg Val Loss: {avg_val_loss:.4f}")
        
        # --- SAVE THE BEST ENCODER WEIGHTS ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # We save the state_dict of the pose_encoder module
            torch.save(model.pose_encoder.state_dict(), BEST_ENCODER_SAVE_PATH)
            print(f"  -> New best model saved to '{BEST_ENCODER_SAVE_PATH}' (Val Loss: {best_val_loss:.4f})")

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

    # --- SAVE THE FINAL ENCODER ---
    # Also save the encoder from the final epoch for reference or resuming
    torch.save(model.pose_encoder.state_dict(), FINAL_ENCODER_SAVE_PATH)
    print(f"\nFinal encoder weights saved to '{FINAL_ENCODER_SAVE_PATH}'")

    print("\nTraining complete.")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title("Training and Validation Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('pose_mae_losses.png', dpi=300)
        print("Loss plot saved to 'pose_mae_losses.png'")
        # plt.show(block=True)
    except ImportError:
        print("\nMatplotlib not found. Skipping plot generation. Install with 'pip install matplotlib'")
