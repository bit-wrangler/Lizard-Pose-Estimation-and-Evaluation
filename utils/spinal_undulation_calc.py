from scipy.signal import find_peaks
import numpy as np
import h5py
import glob
import os
import matplotlib.pyplot as plt
import tqdm
from utils.landmarks import *

PROCESSED_DATA_DIR = 'data/cdl-projects/test1-haag-2025-05-21/processed'

SCALE = 1.75
FPS = 120

from scipy.signal import butter, filtfilt

HP_CUTOFF_HZ = 3.0        # adjust as needed
HP_ORDER = 2

LP_CUTOFF_HZ = 15.0        # adjust as needed
LP_ORDER = 2

MIN_DEFLECTION = 0.5

def _interp_nans_1d(y: np.ndarray) -> np.ndarray:
    """Linear-interpolate NaNs in-place (assumes at least one non-NaN)."""
    nans = np.isnan(y)
    if nans.any():
        x = np.arange(len(y))
        y[nans] = np.interp(x[nans], x[~nans], y[~nans])
    return y

def _butter_highpass(y: np.ndarray, fs: float, fc: float, order: int = 2) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, fc / nyq, btype='high', analog=False)
    return filtfilt(b, a, y, axis=0)

def _butter_lowpass(y: np.ndarray, fs: float, fc: float, order: int = 2) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, fc / nyq, btype='low', analog=False)
    return filtfilt(b, a, y, axis=0)

def get_lizard_base_length(position_data: np.ndarray, min_confidence=0.9):
    """
    Approximates the length of the lizard's body (sans tail) by
    finding the average distance between the snout and spine6 landmarks.

    position_data: (N, 37, 3) array of position data
    """

    snout_positions = position_data[:, SNOUT, :2]
    spine6_positions = position_data[:, SPINE6, :2]
    snout_confidences = position_data[:, SNOUT, 2]
    spine6_confidences = position_data[:, SPINE6, 2]

    valid_mask = (snout_confidences > min_confidence) & (spine6_confidences > min_confidence)
    valid_snout_positions = snout_positions[valid_mask]
    valid_spine6_positions = spine6_positions[valid_mask]

    distances = np.linalg.norm(valid_snout_positions - valid_spine6_positions, axis=1)
    return distances.mean() / SCALE

def get_undulation_amplitude(
        spine1_position_data: np.ndarray, 
        spine3_position_data: np.ndarray, 
        spine4_position_data: np.ndarray, 
        spine6_position_data: np.ndarray, 
        min_confidence_pos=0.6
):
    t = np.linspace(0, spine1_position_data.shape[0] / FPS, spine1_position_data.shape[0])
    spine1_valid_mask = spine1_position_data[:, 2] > min_confidence_pos
    spine3_valid_mask = spine3_position_data[:, 2] > min_confidence_pos
    spine4_valid_mask = spine4_position_data[:, 2] > min_confidence_pos
    spine6_valid_mask = spine6_position_data[:, 2] > min_confidence_pos
    valid_mask = spine1_valid_mask & spine3_valid_mask & spine4_valid_mask & spine6_valid_mask
    # spine1_positions = spine1_position_data[valid_mask, :2]
    # spine3_positions = spine3_position_data[valid_mask, :2]
    # spine4_positions = spine4_position_data[valid_mask, :2]
    # spine6_positions = spine6_position_data[valid_mask, :2]
    # t = t[valid_mask]

    if len(t) < 2:
        return None, None
    
    spine_vector = spine1_position_data[:, :2] - spine6_position_data[:, :2]
    spine_vector = spine_vector / np.linalg.norm(spine_vector, axis=1, keepdims=True)
    spine_vector_perp = np.array([-spine_vector[:, 1], spine_vector[:, 0]]).T
    spine_midpoints = (spine1_position_data[:, :2] + spine6_position_data[:, :2]) / 2.0
    deflected_spine_midpoints = (spine3_position_data[:, :2] + spine4_position_data[:, :2]) / 2.0
    spine_deflection_vector = deflected_spine_midpoints - spine_midpoints
    spine_deflection = np.einsum('ij,ij->i', spine_deflection_vector, spine_vector_perp)


    spine_deflection[~valid_mask] = np.nan
    _interp_nans_1d(spine_deflection)

    spine_deflection_hp = _butter_highpass(spine_deflection, FPS, HP_CUTOFF_HZ, HP_ORDER)
    spine_deflection_lp = _butter_lowpass(spine_deflection_hp, FPS, LP_CUTOFF_HZ, LP_ORDER)

    # drop invalid frames again (optional — keeps downstream logic unchanged)
    spine_deflection_lp[~valid_mask] = np.nan
    spine_deflection = spine_deflection_lp
    # spine_deflection = spine_deflection_lp[valid_mask]
    # t = t[valid_mask]

    return t, spine_deflection / SCALE


if __name__ == "__main__":
    kinematic_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, '*_velocity.h5'))
    video_kinematic_map = {}
    for file in kinematic_files:
        video_id = os.path.basename(file).split('_velocity')[0]
        video_kinematic_map[video_id] = file

    all_rel_peaks = []

    for video_id in tqdm.tqdm(video_kinematic_map):
        # if video_id != '0418_2':
        #     continue
        kinematic_file = video_kinematic_map[video_id]
        with h5py.File(kinematic_file, 'r') as kinematic_f:
            if 'position_filtered_x_y_conf' not in kinematic_f:
                continue
            if 'position_x_y_conf' not in kinematic_f:
                continue
            # position_data = kinematic_f['position_filtered_x_y_conf'][:]
            position_data = kinematic_f['position_x_y_conf'][:]

        lizard_length = get_lizard_base_length(position_data)
        if np.isnan(lizard_length):
            continue

        t, undulation_amplitude = get_undulation_amplitude(
            position_data[:, SPINE1, :],
            position_data[:, SPINE3, :],
            position_data[:, SPINE4, :],
            position_data[:, SPINE6, :]
        )
        if undulation_amplitude is None:
            continue

        peaks, _ = find_peaks(np.abs(undulation_amplitude), height=MIN_DEFLECTION)
        if len(peaks) < 2:
            continue

        avg_peak = np.mean(np.abs(undulation_amplitude[peaks]))
        rel_peak = avg_peak / lizard_length
        # print(f"{video_id}: {avg_peak} ({rel_peak})")
        all_rel_peaks.append(rel_peak)
        
        

        # plt.plot(t, undulation_amplitude)
        # plt.title(video_id)
        # plt.tight_layout()
        # plt.show(block=True)
        # # plt.savefig(f'undulation_amplitudes/{video_id}.png', dpi=300)
        # plt.close()

    print(f"Average relative peak: {np.mean(all_rel_peaks)}")
    print(f"Median relative peak: {np.median(all_rel_peaks)}")
    print(f"Min relative peak: {np.min(all_rel_peaks)}")
    print(f"Max relative peak: {np.max(all_rel_peaks)}")

    plt.hist(all_rel_peaks, bins=50)
    plt.title('Relative Undulation Amplitude')
    plt.tight_layout()
    plt.savefig('relative_undulation_amplitudes.png', dpi=300)
    plt.close()
