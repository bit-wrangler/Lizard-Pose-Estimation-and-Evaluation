import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import h5py
import numpy as np
import os
import glob
import random
from typing import List, Tuple, Dict, Any
import math
import tqdm
import yaml
from pathlib import Path
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve, auc, average_precision_score
import warnings
import pickle
import uuid

warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
# Directories
PROCESSED_DATA_DIR = 'data/cdl-projects/test1-haag-2025-05-21/processed'
ANNOTATIONS_DIR = 'annotations' # Assumed directory for annotation files
# Paths
PRETRAINED_ENCODER_PATH = 'best_pose_encoder.pth'
FINAL_MODEL_SAVE_PATH = 'final_pose_classifier.pth'
# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FREEZE_ENCODER = True

# --- Data Parameters ---
WINDOW_SIZE = 90      # Number of time steps (frames) in each data instance. MUST MATCH MAE.
STRIDE = 5            # Step size to move the window across the time series for classification.
NUM_LANDMARKS = 26    # Number of landmarks in the data. MUST MATCH MAE.
NUM_CHANNELS = 6      # x, y, confidence. MUST MATCH MAE.
ANNOTATION_SET = [
    "left ankle placed",
    "right ankle placed",
    "left wrist placed",
    "right wrist placed",
]
NUM_CLASSES = len(ANNOTATION_SET)

VELOCITY_KEY = 'velocity_filtered_butter_x_y_conf'
POSITION_KEY = 'position_filtered_x_y_conf'

# --- Training Parameters ---
K_FOLDS = 4           # Number of folds for cross-validation.
NUM_EPOCHS = 200       # Number of epochs to train per fold.
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
# Set NUM_WORKERS to 0 if you are on Windows or debugging.
NUM_WORKERS = 10 if DEVICE == "cuda" else 0

# --- Model Hyperparameters ---
# These MUST MATCH the architecture of the saved pre-trained encoder
POSE_EMBEDDING_DIM = 256
POSE_NUM_HEADS = 8
POSE_NUM_LAYERS = 3
POSE_DIM_FEEDFORWARD = 512

TIME_EMBEDDING_DIM = 256
TIME_NUM_HEADS = 8
TIME_NUM_LAYERS = 3
TIME_DIM_FEEDFORWARD = 512

# MLP Head configuration
MLP_HIDDEN_DIM = 512
DROPOUT_RATE = 0.3
CLASSIFIER_HEAD = 'gru' # 'cnn' or 'gru'


from torch.utils.data import Sampler
def make_balanced_sampler(base_ds, idx_list, pos_ratio: float = 0.5):
    """
    base_ds   : the full ClassifierDataset (has .pos_indices / .neg_indices)
    idx_list  : indices that form the current split (train or val)
    """
    subset_pos = [i for i in idx_list if i in base_ds.pos_indices]
    subset_neg = [i for i in idx_list if i in base_ds.neg_indices]

    class _BalancedSampler(Sampler):
        def __init__(self, pos, neg, p_ratio):
            self.pos, self.neg, self.p_ratio = pos, neg, p_ratio

        def __iter__(self):
            n_total = len(self.pos) + len(self.neg)
            n_pos   = max(1, int(n_total * self.p_ratio))
            n_neg   = n_total - n_pos

            # sample with replacement if a pool is too small
            pos_choice = np.random.choice(
                self.pos, n_pos,
                replace=(n_pos > len(self.pos))
            )
            neg_choice = np.random.choice(
                self.neg, n_neg,
                replace=(n_neg > len(self.neg))
            )

            idx = np.concatenate([pos_choice, neg_choice])
            np.random.shuffle(idx)
            return iter(idx.tolist())

        def __len__(self):
            return len(self.pos) + len(self.neg)

    return _BalancedSampler(subset_pos, subset_neg, pos_ratio)

# =====================================================================================
# 1. PRE-TRAINED ENCODER DEFINITION (Copied from MAE for architecture matching)
# =====================================================================================

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

# =====================================================================================
# 2. NEW CLASSIFIER MODEL
# =====================================================================================

class GruClassifierHead(nn.Module):
    def __init__(self, time_dim, num_classes, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(time_dim,
                          time_dim//2,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(time_dim, num_classes)

    def forward(self, x):
        h, _ = self.gru(x)
        h = self.dropout(h)
        return self.classifier(h)

class CnnClassifierHead(nn.Module):
    def __init__(self, time_dim, num_classes, dropout=0.3):
        super().__init__()
        self.cnn = nn.Conv1d(time_dim, time_dim, kernel_size=7, padding=3)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(time_dim, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, T, E) -> (B, E, T)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # (B, E, T) -> (B, T, E)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        return self.classifier(x)

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.001, minimize=True):
        self.patience = patience
        self.min_delta = min_delta
        self.minimize = minimize
        self.counter = 0
        self.best_metric = None
        self.early_stop = False

    def __call__(self, val_metric):
        new_best = False
        metric = val_metric if self.minimize else -val_metric
        if self.best_metric is None:
            self.best_metric = metric
            new_best = True
        elif metric < self.best_metric - self.min_delta:
            self.counter = 0
            self.best_metric = metric
            new_best = True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop, new_best
            
class MultiSignalEarlyStopper:
    def __init__(self, patience=[10], min_delta=[0.001], minimize=[True]):
        self.patience = patience
        self.min_delta = min_delta
        self.minimize = minimize
        self.counter = [0] * len(patience)
        self.best_metric = [None] * len(patience)
        self.early_stop = [False] * len(patience)

    def __call__(self, val_metric:list[float]):
        new_best = False
        assert len(val_metric) == len(self.patience) == len(self.min_delta) == len(self.minimize)
        metrics = [m if self.minimize[i] else -m for i, m in enumerate(val_metric)]
        for i, metric in enumerate(metrics):
            if self.best_metric[i] is None:
                self.best_metric[i] = metric
                new_best = True
            elif metric < self.best_metric[i] - self.min_delta[i]:
                self.counter[i] = 0
                self.best_metric[i] = metric
                new_best = True
            else:
                self.counter[i] += 1
                if self.counter[i] >= self.patience[i]:
                    self.early_stop[i] = True

        return all(self.early_stop), new_best

class PoseClassifier(nn.Module):
    def __init__(self, encoder, time_dim, num_classes, freeze_encoder=True, dropout=0.3):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for p in self.encoder.parameters(): p.requires_grad = False

        if CLASSIFIER_HEAD == 'cnn':
            self.classifier_head = CnnClassifierHead(time_dim, num_classes, dropout)
        else:
            self.classifier_head = GruClassifierHead(time_dim, num_classes, dropout)

    def forward(self, x):          # x: (B,T,L,C)
        B,T,L,C = x.shape
        tokens = x.view(B, T*L, C)
        idx     = torch.arange(T*L, device=x.device).unsqueeze(0).repeat(B,1)
        frame_emb = self.encoder(tokens, idx)        # (B,T,E)

        return self.classifier_head(frame_emb)


# =====================================================================================
# 3. ANNOTATION PARSING AND DATASET
# =====================================================================================

def parse_annotations(annotations_dir: str) -> Dict[str, Dict[int, List[str]]]:
    """Parses all annotation files in a directory."""
    all_annotations = {}
    for file_path in Path(annotations_dir).glob('*.yaml'):
        video_id = file_path.stem
        try:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
                if data:
                    for frame, labels in data.items():
                        if labels:
                            data[frame] = [label.replace('- ', '') for label in labels]
                    all_annotations[video_id] = data
        except Exception as e:
            print(f"Warning: Could not parse annotation file {file_path}. Error: {e}")
    return all_annotations


class ClassifierDataset(Dataset):
    """
    PyTorch Dataset for loading windowed pose data and corresponding multi-label annotations.
    """
    def __init__(self,
                 h5_file_paths: List[str],
                 annotations: Dict[str, Dict[int, List[str]]],
                 window_size: int,
                 stride: int,
                 annotation_set: List[str],
                 dilate: int = 5):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.dilate = dilate
        self.all_annotations = annotations
        self.h5_file_paths = h5_file_paths
        
        self.label_to_idx = {label: i for i, label in enumerate(annotation_set)}
        self.num_classes = len(annotation_set)
        
        self.indices = []
        self._create_indices()
        self.pos_indices = [i for i, (_, s) in enumerate(self.indices)
                    if self._window_has_event(i)]
        self.neg_indices = list(set(range(len(self.indices))) - set(self.pos_indices))

    def _window_has_event(self, idx):
        file_idx, start = self.indices[idx]
        video_id = Path(self.h5_file_paths[file_idx]).stem.split('_velocity')[0]
        anns = self.all_annotations.get(video_id, {})
        # Check if any event annotation exists within the window
        if not anns:
            return False
        return any(start <= frame_num < start + self.window_size for frame_num in anns.keys())


    def _create_indices(self):
        """Pre-calculates indices for every possible window in annotated files."""
        print("Creating dataset indices...")
        for i, file_path in enumerate(tqdm.tqdm(self.h5_file_paths, desc="Processing files")):
            video_id = Path(file_path).stem.split('_velocity')[0]
            if video_id not in self.all_annotations:
                continue

            try:
                with h5py.File(file_path, 'r') as f:
                    n_frames = f[VELOCITY_KEY].shape[0]
                    if n_frames < self.window_size:
                        continue
                    last_start_idx = n_frames - self.window_size
                    for start_idx in range(0, last_start_idx + 1, self.stride):
                        self.indices.append((i, start_idx))
            except Exception as e:
                print(f"Warning: Could not process file {file_path}. Error: {e}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_idx, start_frame = self.indices[idx]
        file_path = self.h5_file_paths[file_idx]
        video_id = Path(file_path).stem.split('_velocity')[0]
        
        with h5py.File(file_path, 'r') as f:
            end_frame = start_frame + self.window_size
            velocity_data = f[VELOCITY_KEY][start_frame:end_frame, :, :]
            position_data = f[POSITION_KEY][start_frame:end_frame, :, :]
            center_frame_pos = position_data[self.window_size // 2, :, :2]
            position_data[:, :, :2] -= center_frame_pos
            data_window = np.concatenate([position_data, velocity_data], axis=2)
        
        data_tensor = torch.from_numpy(data_window).float()
        pos = data_tensor[:, :, :2]
        pos_mean, pos_std = pos.mean(), pos.std().clamp_min(1e-6)
        data_tensor[:, :, :2] = (pos - pos_mean) / pos_std
        
        vel = data_tensor[:, :, 3:5]
        vel_mean, vel_std = vel.mean(), vel.std().clamp_min(1e-6)
        data_tensor[:, :, 3:5] = (vel - vel_mean) / vel_std
        
        video_annotations = self.all_annotations.get(video_id, {})
        labels_tensor = torch.zeros(self.window_size, self.num_classes, dtype=torch.float)
        
        for i in range(self.window_size):
            frame_num = start_frame + i
            if frame_num in video_annotations:
                frame_labels = video_annotations[frame_num]
                if frame_labels: 
                    for label in frame_labels:
                        if label in self.label_to_idx:
                            label_idx = self.label_to_idx[label]
                            labels_tensor[i, label_idx] = 1.0

        dilated = labels_tensor.clone()
        for t in range(self.window_size):
            if labels_tensor[t].any():
                lo = max(0, t)
                hi = min(self.window_size, t + self.dilate)
                dilated[lo:hi] = torch.maximum(dilated[lo:hi], labels_tensor[t])
        labels_tensor = dilated

        return data_tensor, labels_tensor

# =====================================================================================
# 4. TRAINING, VALIDATION, AND INFERENCE FUNCTIONS
# =====================================================================================

def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    
    pbar = tqdm.tqdm(dataloader, desc="Training", leave=False)
    for data, targets in pbar:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(data)
        loss = loss_fn(logits.view(-1, NUM_CLASSES), targets.view(-1, NUM_CLASSES))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    return total_loss / len(dataloader)

def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_targets = [], []
    
    pbar = tqdm.tqdm(dataloader, desc="Validating", leave=False)
    with torch.no_grad():
        for data, targets in pbar:
            data, targets = data.to(device), targets.to(device)
            logits = model(data)
            loss = loss_fn(logits.view(-1, NUM_CLASSES), targets.view(-1, NUM_CLASSES))
            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            all_probs.append(probs.view(-1, NUM_CLASSES).cpu())
            all_targets.append(targets.view(-1, NUM_CLASSES).cpu())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    binary_preds = (all_probs > 0.5)
    f1 = f1_score(all_targets, binary_preds, average='samples', zero_division=0)
    accuracy = accuracy_score(all_targets, binary_preds)
    prauc = average_precision_score(all_targets, all_probs, average="samples")
    event_prec, event_rec, event_f1_val = event_f1(all_probs, all_targets, thresh=0.5, tol=3)
    
    print(f"Event-level  P:{event_prec:.3f}  R:{event_rec:.3f}  F1:{event_f1_val:.3f}")

    return avg_loss, f1, accuracy, prauc, all_probs, all_targets, event_f1_val

def event_f1(all_probs: np.ndarray,
             all_targets: np.ndarray,
             thresh: float = 0.5,
             tol: int = 3) -> Tuple[float, float, float]:
    num_classes = all_probs.shape[1]
    tot_tp = tot_fp = tot_fn = 0

    for c in range(num_classes):
        preds  = np.where(all_probs[:, c] > thresh)[0]
        gts    = np.where(all_targets[:, c] == 1)[0]
        matched_gt = np.zeros(len(gts), dtype=bool)
        for p in preds:
            ok = np.where(np.abs(gts - p) <= tol)[0]
            ok = ok[~matched_gt[ok]]
            if ok.size:
                matched_gt[ok[0]] = True
                tot_tp += 1
            else:
                tot_fp += 1
        tot_fn += (~matched_gt).sum()

    prec = tot_tp / (tot_tp + tot_fp + 1e-8)
    rec  = tot_tp / (tot_tp + tot_fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    return prec, rec, f1

# =====================================================================================
# START: Modified function for inference on validation files
# =====================================================================================
def perform_inference_on_validation_fold(model: nn.Module,
                                         h5_file_handle: h5py.File,
                                         validation_files: List[str],
                                         device: str,
                                         window_size: int,
                                         stride: int,
                                         num_classes: int):
    """
    Applies the trained model to full validation files using a sliding window and saves
    the averaged probabilities to an HDF5 file handle.
    """
    model.eval()
    print(f"Performing inference on {len(validation_files)} validation files for this fold...")

    for file_path in tqdm.tqdm(validation_files, desc="Inference on validation set"):
        video_id = Path(file_path).stem.split('_velocity')[0]

        try:
            with h5py.File(file_path, 'r') as f:
                velocity_data = f[VELOCITY_KEY][:]
                position_data = f[POSITION_KEY][:]
                n_frames = velocity_data.shape[0]

            sum_probs = np.zeros((n_frames, num_classes), dtype=np.float32)
            count_probs = np.zeros((n_frames, num_classes), dtype=np.int32)

            for start_frame in range(0, n_frames - window_size + 1, stride):
                end_frame = start_frame + window_size
                
                vel_window = velocity_data[start_frame:end_frame]
                pos_window = position_data[start_frame:end_frame]

                center_frame_pos = pos_window[window_size // 2, :, :2]
                pos_window_centered = pos_window.copy()
                pos_window_centered[:, :, :2] -= center_frame_pos
                
                data_window = np.concatenate([pos_window_centered, vel_window], axis=2)
                data_tensor = torch.from_numpy(data_window).float()
                
                pos = data_tensor[:, :, :2]
                pos_mean, pos_std = pos.mean(), pos.std().clamp_min(1e-6)
                data_tensor[:, :, :2] = (pos - pos_mean) / pos_std
                
                vel = data_tensor[:, :, 3:5]
                vel_mean, vel_std = vel.mean(), vel.std().clamp_min(1e-6)
                data_tensor[:, :, 3:5] = (vel - vel_mean) / vel_std

                data_tensor = data_tensor.unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(data_tensor)
                    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

                sum_probs[start_frame:end_frame] += probs
                count_probs[start_frame:end_frame] += 1
            
            count_probs[count_probs == 0] = 1
            final_avg_probs = sum_probs / count_probs
            
            h5_file_handle.create_dataset(video_id, data=final_avg_probs)

        except Exception as e:
            print(f"\nWarning: Could not run inference on file {file_path}. Error: {e}")
# =====================================================================================
# END: Modified function
# =====================================================================================


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = p*targets + (1-p)*(1-targets)
        loss = ce * (self.alpha*(1-p_t)**self.gamma)
        return loss.mean()

# =====================================================================================
# 5. MAIN EXECUTION BLOCK
# =====================================================================================

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print("-" * 50)

    annotations = parse_annotations(ANNOTATIONS_DIR)
    if not annotations:
        raise ValueError(f"No annotations found or parsed in '{ANNOTATIONS_DIR}'.")
    
    all_h5_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, '*.h5'))
    annotated_h5_files = sorted([f for f in all_h5_files if Path(f).stem.split('_velocity')[0] in annotations])
    
    if not annotated_h5_files:
        raise FileNotFoundError("No HDF5 files in PROCESSED_DATA_DIR match the annotation files.")
        
    print(f"Found {len(annotations)} videos with annotations.")
    print(f"Found {len(annotated_h5_files)} matching HDF5 files.")
    print("-" * 50)
    
    base_encoder = PoseEncoder(
        window_size=WINDOW_SIZE, num_landmarks=NUM_LANDMARKS, num_channels=NUM_CHANNELS,
        pose_embedding_dim=POSE_EMBEDDING_DIM, pose_num_heads=POSE_NUM_HEADS,
        pose_num_layers=POSE_NUM_LAYERS, pose_dim_feedforward=POSE_DIM_FEEDFORWARD,
        temporal_stage_embedding_dim=TIME_EMBEDDING_DIM, time_num_heads=TIME_NUM_HEADS,
        time_num_layers=TIME_NUM_LAYERS, time_dim_feedforward=TIME_DIM_FEEDFORWARD
    )
    
    try:
        base_encoder.load_state_dict(torch.load(PRETRAINED_ENCODER_PATH, map_location=DEVICE))
        print(f"Successfully loaded pre-trained encoder from '{PRETRAINED_ENCODER_PATH}'")
    except FileNotFoundError:
        print(f"WARNING: Pre-trained encoder not found at '{PRETRAINED_ENCODER_PATH}'. The encoder will be trained from scratch.")
    except Exception as e:
        print(f"ERROR: Could not load encoder weights. Check architecture. Error: {e}")
        exit()
    print("-" * 50)

    dataset = ClassifierDataset(
        h5_file_paths=annotated_h5_files,
        annotations=annotations,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        annotation_set=ANNOTATION_SET
    )

    groups = [Path(dataset.h5_file_paths[file_idx]).stem.split('_velocity')[0] for file_idx, _ in dataset.indices]
    
    n_folds = min(len(np.unique(groups)), K_FOLDS)
    gkf   = GroupKFold(n_splits=n_folds)
    folds = gkf.split(X=np.arange(len(dataset)), y=np.zeros(len(dataset)), groups=groups)
    
    fold_results = []
    all_fold_metrics = []
    run_id = str(uuid.uuid4())
    
    # =====================================================================================
    # START: Modified section for single aggregated HDF5 file
    # =====================================================================================
    inference_filename = f'all_folds_inference_results_{run_id}.h5'
    print(f"Starting {n_folds}-Fold Cross-Validation. Run ID: {run_id}")
    print(f"Aggregated inference results will be saved to '{inference_filename}'")

    with h5py.File(inference_filename, 'w') as inference_hf:
        for fold, (train_ids, val_ids) in enumerate(folds):
            print(f"\n===== FOLD {fold+1}/{n_folds} =====")
            fold_metrics = []
            best_val_metric = [-1, -1] # PRAUC, Event F1
            
            train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=make_balanced_sampler(dataset, train_ids, 0.5), num_workers=NUM_WORKERS)
            val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=make_balanced_sampler(dataset, val_ids, 0.5), num_workers=NUM_WORKERS)
            
            model = PoseClassifier(
                encoder=base_encoder, time_dim=TIME_EMBEDDING_DIM,
                num_classes=NUM_CLASSES, freeze_encoder=FREEZE_ENCODER,
                dropout=DROPOUT_RATE
            ).to(DEVICE)
            
            if not FREEZE_ENCODER:
                for p in model.encoder.parameters(): p.requires_grad = True
                optimizer = torch.optim.AdamW([
                    {"params": model.encoder.parameters(), "lr": LEARNING_RATE * 0.1},
                    {"params": model.classifier_head.parameters()}], lr=LEARNING_RATE)
            else:
                optimizer = torch.optim.AdamW(model.classifier_head.parameters(), lr=LEARNING_RATE)
            
            loss_fn = FocalLoss()
            early_stopper = MultiSignalEarlyStopper(minimize=[True,False,False], patience=[25,25,25], min_delta=[0.001,0.001,0.001])

            for epoch in range(NUM_EPOCHS):
                train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
                val_loss, val_f1, val_acc, val_prauc, _, _, val_event_f1 = validate(model, val_loader, loss_fn, DEVICE)
                
                fold_metrics.append([train_loss, val_loss, val_prauc, val_event_f1])
                early_stop, new_best = early_stopper([val_loss, val_prauc, val_event_f1])
                
                if new_best:
                    print(f"New best model (Val PRAUC: {val_prauc:.4f}, Val Event F1: {val_event_f1:.4f})")
                    best_val_metric = [val_prauc, val_event_f1]
            
                print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PRAUC: {val_prauc:.4f} | Val Event F1: {val_event_f1:.4f}")
                
                if early_stop:
                    print("Early stopping triggered.")
                    break

            fold_results.append(best_val_metric)
            all_fold_metrics.append(fold_metrics)
            print(f"Fold {fold+1} Best Val PRAUC: {best_val_metric[0]:.4f}, Event F1: {best_val_metric[1]:.4f}")

            val_file_indices = sorted(list(set([dataset.indices[i][0] for i in val_ids])))
            val_h5_files = [dataset.h5_file_paths[i] for i in val_file_indices]
            
            perform_inference_on_validation_fold(
                model=model,
                h5_file_handle=inference_hf,
                validation_files=val_h5_files,
                device=DEVICE,
                window_size=WINDOW_SIZE,
                stride=STRIDE,
                num_classes=NUM_CLASSES
            )
    # =====================================================================================
    # END: Modified section
    # =====================================================================================

    event_f1_scores = [f[1] for f in fold_results]
    prauc_scores = [f[0] for f in fold_results]

    print("\n===== Cross-Validation Summary =====")
    print(f"Event F1-Scores per fold: {[f'{f:.4f}' for f in event_f1_scores]}")
    print(f"Average Event F1-Score: {np.mean(event_f1_scores):.4f} (+/- {np.std(event_f1_scores):.4f})")
    print(f"PRAUC per fold: {[f'{f:.4f}' for f in prauc_scores]}")
    print(f"Average PRAUC: {np.mean(prauc_scores):.4f} (+/- {np.std(prauc_scores):.4f})")
    print("-" * 50)

    run_results = {
        'event_f1_scores': event_f1_scores,
        'prauc_scores': prauc_scores,
        'fold_metrics': [np.array(f) for f in all_fold_metrics],
        'metadata': {
            'run_id': run_id,
            'NUM_EPOCHS': NUM_EPOCHS, 'BATCH_SIZE': BATCH_SIZE, 'LEARNING_RATE': LEARNING_RATE,
            'POSE_EMBEDDING_DIM': POSE_EMBEDDING_DIM, 'TIME_EMBEDDING_DIM': TIME_EMBEDDING_DIM,
            'DROPOUT_RATE': DROPOUT_RATE, 'WINDOW_SIZE': WINDOW_SIZE, 'STRIDE': STRIDE,
            'CLASSIFIER_HEAD': CLASSIFIER_HEAD, 'n_folds': n_folds
        }
    }

    results_filename_base = f'cv_results_{run_id}'
    with open(f'{results_filename_base}.pkl', 'wb') as f:
        pickle.dump(run_results, f)
    with open(f'{results_filename_base}.yaml', 'w') as f:
        yaml.dump(run_results['metadata'], f, default_flow_style=False)
    
    print(f"Saved cross-validation metrics and metadata to '{results_filename_base}.pkl/.yaml'")