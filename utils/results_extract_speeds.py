import h5py
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow as pa
import tqdm
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from scipy.signal import lfilter, butter, filtfilt


results_root = 'data/cdl-projects/test1-haag-2025-05-21/videos'
output_dir = 'data/cdl-projects/test1-haag-2025-05-21/processed'

SNOUT_MIN_CONFIDENCE = 0.95


SCALE = 1.75 # px/mm
VIDEO_FPS = 120
VELOCITY_FILTER_CUTOFF = 15

def apply_filter(data):
    """Applies a low-pass filter to the velocity data."""
    nyquist = 0.5 * VIDEO_FPS
    cutoff = VELOCITY_FILTER_CUTOFF
    # normal_cutoff = VELOCITY_FILTER_CUTOFF / nyquist
    b, a = butter(2, cutoff, btype='low', analog=False, fs=VIDEO_FPS)
    T, L, C = data.shape
    flat = data.reshape(T, -1)          # (T, L*C)
    filtered = filtfilt(b, a, flat, axis=0)
    return filtered.reshape(T, L, C)

def ema_iir(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Exponential-moving-average along the time axis (axis 0).
    x shape: (N, L, 2)
    """
    b = [alpha]                   # numerator  (order-0)
    a = [1, -(1 - alpha)]         # denominator (order-1)
    y, _ = lfilter(
        b, a, x,
        axis=0,
        zi=x[:1] * (1 - alpha)    # warm-start to avoid zero-bias
    )
    return y                      # same (N, L, 2) shape

def get_velocity_data(data):

    n = data.shape[0]
    
    snout_start = None
    snout_end = None
    
    for t in range(1, n):
        if data[t, 0, 2] < SNOUT_MIN_CONFIDENCE:
            snout_start = data[t-1, 0, :2]
            break
    for t in range(n-1, 0, -1):
        if data[t, 0, 2] < SNOUT_MIN_CONFIDENCE:
            snout_end = data[t, 0, :2]
            break

    if snout_start is None or snout_end is None:
        return None

    full_video_snout_dx = snout_end - snout_start
    full_video_snout_dx_unit_vector = full_video_snout_dx / np.linalg.norm(full_video_snout_dx)

    # rotate 90 degrees
    full_video_snout_dy_unit_vector = np.array([full_video_snout_dx_unit_vector[1], -full_video_snout_dx_unit_vector[0]])

    # calculate frame-to-frame diff for all landmarks
    landmark_diff = np.diff(data[:, :, :2], axis=0)
    landmark_diff = np.concatenate([landmark_diff, landmark_diff[-1:, :, :]], axis=0)

    # project onto unit vectors
    landmark_dx = (landmark_diff * full_video_snout_dx_unit_vector).sum(axis=2).reshape(n, 26, 1)
    landmark_dy = (landmark_diff * full_video_snout_dy_unit_vector).sum(axis=2).reshape(n, 26, 1)

    mean_diff_confidence = (data[:, :, 2] + np.concatenate([data[1:, :, 2], np.zeros((1, 26))], axis=0)) / 2
    mean_diff_confidence = np.nan_to_num(mean_diff_confidence, nan=0.0, posinf=0.0, neginf=0.0)
    mean_diff_confidence = mean_diff_confidence.reshape(n, 26, 1)

    multiplicative_confidence = (data[:, :, 2] * np.concatenate([data[1:, :, 2], np.zeros((1, 26))], axis=0)).reshape(n, 26, 1)

    velocity = np.concatenate([landmark_dx, landmark_dy, multiplicative_confidence], axis=2)
    velocity[:, :, :2] = velocity[:, :, :2] / SCALE * VIDEO_FPS
    return velocity
    
def get_position_data(result_file):
    with h5py.File(result_file, 'r') as f:
        data = np.array(list(map(lambda x: x[1], f['df_with_missing']['table'])))
        n = data.shape[0]
        data = data.reshape(n, 26, 3)
        return data
    
def get_confidence_filtered_position_data(position):
    out = np.zeros_like(position)
    out[:, :, 2] = position[:, :, 2]
    out[0, :, :2] = position[0, :, :2]
    for t in range(1, position.shape[0]):
        alpha = position[t, :, 2]
        out[t, :, :2] = position[t, :, :2] * alpha.reshape(-1, 1) + out[t-1, :, :2] * (1 - alpha.reshape(-1, 1))
    return out

if __name__ == "__main__":
    files = glob.glob(os.path.join(results_root, '**', '*DLC_*shuffle*.h5'), recursive=True)#[:100]
    all_velocity_data = []
    pbar = tqdm.tqdm(files, total=len(files), desc="Processing files", position=0, leave=True)
    for result_file in pbar:
        filename = os.path.basename(result_file).split('DLC_')[0]
        specimen = filename.split('_')[0]
        pbar.set_description(f"Processing {filename}")
        position = get_position_data(result_file)
        position_filtered = get_confidence_filtered_position_data(position)
        velocity = get_velocity_data(position)
        if velocity is None: continue
        velocity_filtered = get_velocity_data(position_filtered)
        if velocity_filtered is None: continue
        velocity_filtered_butter = velocity_filtered.copy()
        velocity_filtered_butter[:, :, :2] = apply_filter(velocity_filtered[:, :, :2])
        # apply low pass filter
        raw_velocity = velocity.copy()
        raw_velocity[:, :, :2] = raw_velocity[:, :, :2] / 1000
        raw_velocity[:, :, :2] = np.clip(raw_velocity[:, :, :2], -5000, 5000)

        velocity_filtered[:, :, :2] = np.clip(velocity_filtered[:, :, :2], -5000, 5000)
        velocity_filtered[:, :, :2] = velocity_filtered[:, :, :2] / 1000

        velocity_filtered_butter[:, :, :2] = np.clip(velocity_filtered_butter[:, :, :2], -5000, 5000)
        velocity_filtered_butter[:, :, :2] = velocity_filtered_butter[:, :, :2] / 1000

        # plt.plot(raw_velocity[:, 0, 0], label='raw_x')
        # plt.plot(velocity_filtered[:, 0, 0], label='filtered_x')
        # plt.plot(velocity_filtered_butter[:, 0, 0], label='butter_x')
        # plt.legend()
        # plt.show(block=True)

        velocity[:, :, :2] = ema_iir(velocity[:, :, :2], alpha=0.33)
        velocity[:, :, :2] = np.clip(velocity[:, :, :2], -5000, 5000)
        velocity[:, :, :2] = velocity[:, :, :2] / 1000
        output_file = os.path.join(output_dir, f'{filename}_velocity.h5')
        data = {
            'velocity_x_y_conf': velocity,
            'raw_velocity_x_y_conf': raw_velocity,
            'position_x_y_conf': position,
            'position_filtered_x_y_conf': position_filtered,
            'velocity_filtered_x_y_conf': velocity_filtered,
            'velocity_filtered_butter_x_y_conf': velocity_filtered_butter,
            # 'median': all_velocity_median,
            # 'iqr': all_velocity_iqr
        }
        with h5py.File(output_file, 'w') as f:
            for k, v in data.items():
                f.create_dataset(k, data=v)
        # all_velocity_data.append(velocity)

    exit()

    all_velocity_data = np.concatenate(all_velocity_data, axis=0)
    all_velocity_data[:, :, :2] = np.clip(all_velocity_data[:, :, :2], -5000, 5000)
    # all_velocity_median = np.median(all_velocity_data[:, :, :2], axis=0)
    # all_velocity_iqr = np.percentile(all_velocity_data[:, :, :2], 75, axis=0) - np.percentile(all_velocity_data[:, :, :2], 25, axis=0)
    # all_velocity_std = np.std(all_velocity_data[:, :, :2], axis=0)

    # all_velocity_snout_x = all_velocity_data[:, 0, 0]
    # all_velocity_snout_x = np.clip(all_velocity_snout_x, -5000, 5000)

    scaler = QuantileTransformer(output_distribution='normal', random_state=0)
    scaler.fit(all_velocity_data[:, :, :2].reshape(all_velocity_data.shape[0],all_velocity_data.shape[1]*2))
    # all_velocity_snout_x_scaled = scaler.fit_transform(all_velocity_snout_x.reshape(-1, 1)).flatten()
    # all_velocity_snout_x_inverse = scaler.inverse_transform(all_velocity_snout_x_scaled.reshape(-1, 1)).flatten()
    # all_velocity_snout_x_inverse_diff = all_velocity_snout_x_inverse - all_velocity_snout_x
    
    # plt.hist(all_velocity_snout_x_scaled, bins=50, density=True)
    # # plt.vlines(all_velocity_median[0][0], 0, 0.02, colors='r', linestyles='dashed')
    # # plt.vlines(all_velocity_median[0][0] + all_velocity_iqr[0][0], 0, 0.02, colors='r', linestyles='dashed')
    # # plt.vlines(all_velocity_median[0][0] - all_velocity_iqr[0][0], 0, 0.02, colors='r', linestyles='dashed')
    # # plt.vlines(all_velocity_median[0][0] + 2*all_velocity_iqr[0][0], 0, 0.02, colors='b', linestyles='dashed')
    # # plt.vlines(all_velocity_median[0][0] - 2*all_velocity_iqr[0][0], 0, 0.02, colors='b', linestyles='dashed')
    # plt.title('Snout Velocity X')
    # plt.tight_layout()
    # plt.show(block=True)
    # plt.close()

    pbar = tqdm.tqdm(files, total=len(files), desc="Processing files", position=0, leave=True)
    os.makedirs(output_dir, exist_ok=True)
    for result_file in pbar:
        filename = os.path.basename(result_file).split('DLC_')[0]
        specimen = filename.split('_')[0]
        pbar.set_description(f"Processing {filename}")
        velocity = get_velocity_data(result_file)
        if velocity is None: continue
        velocity[:, :, :2] = np.clip(velocity[:, :, :2], -5000, 5000)
        velocity[:, :, :2] = scaler.transform(velocity[:, :, :2].reshape(velocity.shape[0],velocity.shape[1]*2)).reshape(velocity.shape[0],velocity.shape[1],2)
        # velocity[:, :, :2] = np.clip(velocity[:, :, :2], -3, 3)
        output_file = os.path.join(output_dir, f'{filename}_velocity.h5')
        data = {
            'velocity_x_y_conf': velocity,
            # 'median': all_velocity_median,
            # 'iqr': all_velocity_iqr
        }
        with h5py.File(output_file, 'w') as f:
            for k, v in data.items():
                f.create_dataset(k, data=v)
