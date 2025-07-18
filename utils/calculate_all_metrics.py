from scipy.signal import find_peaks
import numpy as np
import h5py
import glob
import os
import matplotlib.pyplot as plt
import tqdm
from landmarks import *
from collections import defaultdict
from scipy.signal import butter, filtfilt
import pandas as pd

"""
This script calculates the following metrics for all videos with results:
- Lizard length without tail (from snout to spine6)
- Maximum speed
- Maximum stride length
- Maximum undulation amplitude

"""

PROCESSED_DATA_DIR = 'data/cdl-projects/test1-haag-2025-05-21/processed'
STRIDE_EVENTS_H5_PATH = os.path.join(PROCESSED_DATA_DIR, 'stride_events.h5')
STRIDE_EVENT_THRESHOLD = 0.5
ANNOTATION_SET = [
    "left ankle placed",
    "right ankle placed",
    "left wrist placed",
    "right wrist placed",
]
FPS = 120
SCALE = 1.75 # px/mm

def _interp_nans_2d(y: np.ndarray) -> np.ndarray:
    """Linear-interpolate NaNs in-place (assumes at least one non-NaN)."""
    for i in range(y.shape[1]):
        nans = np.isnan(y[:,i])
        if nans.any():
            x = np.arange(len(y))
            y[nans,i] = np.interp(x[nans], x[~nans], y[~nans,i])
    return y

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


def get_lizard_base_length(
        position_data: np.ndarray, 
        min_confidence=0.9, 
        min_spine_confidence=0.6,
        max_relative_deviation=0.05,
        x_valid_range=(1920//2 - 200, 1920//2 + 200)
        ):
    """
    Approximates the length of the lizard's body (sans tail) by
    finding the average distance between the snout and spine6 landmarks.
    The measurements are taken from frames where both landmarks are confidently detected.
    The measurements are taken from frames where base of head and other spine landmarks are
    also confidently detected and aligned with the snout-spine6 line.
    The measurements are taken from frames where the snout is within the valid x range (close to center of frame).

    position_data: (N, 37, 3) array of position data
    """

    snout_positions = position_data[:, SNOUT, :2]
    spine6_positions = position_data[:, SPINE6, :2]
    snout_confidences = position_data[:, SNOUT, 2]
    spine6_confidences = position_data[:, SPINE6, 2]
    initial_candidate_mask = (snout_confidences > min_confidence) & (spine6_confidences > min_confidence)
    candidate_mask = initial_candidate_mask

    candidate_mask = candidate_mask & (snout_positions[:, 0] > x_valid_range[0]) & (snout_positions[:, 0] < x_valid_range[1])

    snout_spine6_vectors = snout_positions - spine6_positions
    snout_spine6_unit_vectors = snout_spine6_vectors / np.linalg.norm(snout_spine6_vectors, axis=1, keepdims=True)
    snout_spine6_perpendicular_vectors = np.array([-snout_spine6_unit_vectors[:, 1], snout_spine6_unit_vectors[:, 0]]).T 
    distances = np.linalg.norm(snout_spine6_vectors, axis=1)

    landmarks = [
        BASE_OF_HEAD,
        SPINE1,
        SPINE2,
        SPINE3,
        SPINE4,
        SPINE5,
    ]

    for landmark in landmarks:
        if not np.any(candidate_mask):
            return None
        positions = position_data[:, landmark, :2]
        confidences = position_data[:, landmark, 2]
        candidate_mask = candidate_mask & (confidences > min_spine_confidence)
        positions_relative_to_spine6 = positions - spine6_positions
        perpendicular_deviation = np.abs(np.einsum('ij,ij->i', positions_relative_to_spine6, snout_spine6_perpendicular_vectors))
        candidate_mask = candidate_mask & (perpendicular_deviation < max_relative_deviation * distances)

    return distances[candidate_mask].mean() if np.any(candidate_mask) else None

def get_lizard_max_speed(
        position_data,
        min_confidence=0.9,
        low_pass_cutoff_hz=4,
        low_pass_order=2
):
    snout_positions = position_data[:, SNOUT, :2].copy()
    snout_confidences = position_data[:, SNOUT, 2]
    valid_mask = snout_confidences > min_confidence
    snout_positions[~valid_mask] = np.nan

    if not np.any(valid_mask):
        return None

    _interp_nans_2d(snout_positions)
    snout_diff = np.diff(snout_positions, axis=0)
    snout_diff = np.concatenate([np.zeros((1, 2)), snout_diff], axis=0)
    snout_speed = np.linalg.norm(snout_diff, axis=1)
    snout_speed = _butter_lowpass(snout_speed, FPS, low_pass_cutoff_hz, low_pass_order)
    return snout_speed.max() * FPS

def get_limb_stride_length(
        limb_position_data: np.ndarray, 
        limb_events: np.ndarray, 
        body_length: float,
        min_confidence_pos=0.7, min_confidence_event=0.5):
    """
    Estimates the stride length by finding the average distance between
    consecutive peaks in the limb event data.

    limb_position_data: (N, 3) array of position data for the limb of interest
    limb_events: (N, 1) array of limb event data
    """
    event_peaks, _ = find_peaks(limb_events, height=min_confidence_event)
    # sort the peak indices
    event_peaks = np.sort(event_peaks)
    # find the distance between consecutive peaks
    distances = np.linalg.norm(limb_position_data[event_peaks[1:], :2] - limb_position_data[event_peaks[:-1], :2], axis=1)
    confidences = limb_position_data[event_peaks[1:], 2] * limb_position_data[event_peaks[:-1], 2]
    valid_mask = confidences > min_confidence_pos
    valid_mask = valid_mask & (distances / SCALE > 10.0) & (distances < body_length * 6.0)
    return distances[valid_mask].max() if np.any(valid_mask) else None
    
def get_stride_length(
        position_data: np.ndarray,
        events: np.ndarray,
        body_length: float,
        min_confidence_pos=0.7, min_confidence_event=0.5
):
    left_ankle_stride_length = get_limb_stride_length(
                position_data[:, LEFT_ANKLE, :],
                events[:, ANNOTATION_SET.index("left ankle placed")],
                body_length,
                min_confidence_pos,
                min_confidence_event
            )
    right_ankle_stride_length = get_limb_stride_length(
        position_data[:, RIGHT_ANKLE, :],
        events[:, ANNOTATION_SET.index("right ankle placed")],
        body_length,
        min_confidence_pos,
        min_confidence_event
    )

    any_valid = (
                left_ankle_stride_length is not None or
                right_ankle_stride_length is not None 
            )
    
    max_leg_stride = -1
    if left_ankle_stride_length is not None:
        max_leg_stride = max(max_leg_stride, left_ankle_stride_length)
    if right_ankle_stride_length is not None:
        max_leg_stride = max(max_leg_stride, right_ankle_stride_length)
    
    return max_leg_stride if any_valid else None

def get_undulation_amplitude(
        position_data: np.ndarray,
        min_confidence_pos=0.6,
        low_pass_cutoff_hz=15.0,
        low_pass_order=2,
        high_pass_cutoff_hz=2.0,
        high_pass_order=2,
        min_deflection=0.5
):
    spine1_position_data = position_data[:, SPINE1, :]
    spine3_position_data = position_data[:, SPINE3, :]
    spine4_position_data = position_data[:, SPINE4, :]
    spine6_position_data = position_data[:, SPINE6, :]

    t = np.linspace(0, spine1_position_data.shape[0] / FPS, spine1_position_data.shape[0])
    spine1_valid_mask = spine1_position_data[:, 2] > min_confidence_pos
    spine3_valid_mask = spine3_position_data[:, 2] > min_confidence_pos
    spine4_valid_mask = spine4_position_data[:, 2] > min_confidence_pos
    spine6_valid_mask = spine6_position_data[:, 2] > min_confidence_pos
    valid_mask = spine1_valid_mask & spine3_valid_mask & spine4_valid_mask & spine6_valid_mask

    if len(t) < 2:
        return None
    
    spine_vector = spine1_position_data[:, :2] - spine6_position_data[:, :2]
    spine_vector = spine_vector / np.linalg.norm(spine_vector, axis=1, keepdims=True)
    spine_vector_perp = np.array([-spine_vector[:, 1], spine_vector[:, 0]]).T
    spine_midpoints = (spine1_position_data[:, :2] + spine6_position_data[:, :2]) / 2.0
    deflected_spine_midpoints = (spine3_position_data[:, :2] + spine4_position_data[:, :2]) / 2.0
    spine_deflection_vector = deflected_spine_midpoints - spine_midpoints
    spine_deflection = np.einsum('ij,ij->i', spine_deflection_vector, spine_vector_perp)

    if not np.any(valid_mask):
        return None

    spine_deflection[~valid_mask] = np.nan
    _interp_nans_1d(spine_deflection)

    spine_deflection_hp = _butter_highpass(spine_deflection, FPS, high_pass_cutoff_hz, high_pass_order)
    spine_deflection_lp = _butter_lowpass(spine_deflection_hp, FPS, low_pass_cutoff_hz, low_pass_order)

    spine_deflection_lp[~valid_mask] = np.nan
    spine_deflection = spine_deflection_lp

    peaks, _ = find_peaks(np.abs(spine_deflection), height=min_deflection)
    if len(peaks) < 2:
        return None
    
    avg_peak = np.mean(np.abs(spine_deflection[peaks]))
    return avg_peak

if __name__ == "__main__":
    kinematic_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, '*_velocity.h5'))
    video_kinematic_map = {}
    for file in kinematic_files:
        video_id = os.path.basename(file).split('_velocity')[0]
        video_kinematic_map[video_id] = file

    results = defaultdict(dict)

    all_lengths = []
    all_speeds = []
    all_normalized_speeds = []
    all_normalized_stride_lengths = []
    all_normalized_undulation_amplitudes = []

    with h5py.File(STRIDE_EVENTS_H5_PATH, 'r') as f_events:
        all_events = {video_id: f_events[video_id][:] for video_id in f_events.keys()}

    for video_id in tqdm.tqdm(video_kinematic_map):
        lizard_id = video_id.split('_')[0]
        if len(lizard_id) != 4:
            continue
        with h5py.File(video_kinematic_map[video_id], 'r') as f:
            if 'position_x_y_conf' not in f:
                continue
            if 'velocity_x_y_conf' not in f:
                continue
            position_data = f['position_x_y_conf'][:]
            velocity_data = f['velocity_x_y_conf'][:] * SCALE
        results[video_id]['video_id'] = video_id
        results[video_id]['lizard_id'] = lizard_id

        lizard_max_speed = get_lizard_max_speed(position_data)
        results[video_id]['lizard_max_speed_pixels'] = lizard_max_speed
        if lizard_max_speed is not None:
            all_speeds.append(lizard_max_speed)

        lizard_length = get_lizard_base_length(position_data)
        results[video_id]['lizard_length_pixels'] = lizard_length
        if lizard_length is not None:
            all_lengths.append(lizard_length)

        results[video_id]['lizard_max_speed_normalized'] = None
        if lizard_length is not None and lizard_max_speed is not None:
            results[video_id]['lizard_max_speed_normalized'] = lizard_max_speed / lizard_length
            all_normalized_speeds.append(lizard_max_speed / lizard_length)

        if lizard_length is not None:
            if video_id in all_events:
                events = all_events[video_id]
                stride_length = get_stride_length(position_data, events, lizard_length)
                results[video_id]['stride_length_pixels'] = stride_length
                if stride_length is not None:
                    results[video_id]['stride_length_normalized'] = stride_length / lizard_length
                    all_normalized_stride_lengths.append(stride_length / lizard_length)

        spine_undulation_amplitude = get_undulation_amplitude(position_data)
        results[video_id]['spine_undulation_amplitude_pixels'] = spine_undulation_amplitude
        if spine_undulation_amplitude is not None and lizard_length is not None:
            results[video_id]['spine_undulation_amplitude_normalized'] = spine_undulation_amplitude / lizard_length
            all_normalized_undulation_amplitudes.append(spine_undulation_amplitude / lizard_length)
                
    df = pd.DataFrame(results.values())
    df.sort_values(by='lizard_id', inplace=True)
    df.to_csv('all_metrics.csv', index=False)

    # exit()

    plt.hist(all_lengths, bins=50)
    plt.title('Lizard Length')
    plt.tight_layout()
    # plt.savefig('lizard_lengths.png', dpi=300)
    plt.show(block=True)
    plt.close()

    plt.hist(all_speeds, bins=50)
    plt.title('Lizard Max Speed')
    plt.tight_layout()
    # plt.savefig('lizard_max_speeds.png', dpi=300)
    plt.show(block=True)
    plt.close()

    plt.hist(all_normalized_speeds, bins=50)
    plt.title('Lizard Max Speed / Length')
    plt.tight_layout()
    # plt.savefig('lizard_max_speeds_normalized.png', dpi=300)
    plt.show(block=True)
    plt.close()
        
    plt.hist(all_normalized_stride_lengths, bins=50)
    plt.title('Stride Length / Length')
    plt.tight_layout()
    # plt.savefig('stride_lengths_normalized.png', dpi=300)
    plt.show(block=True)
    plt.close()

    plt.hist(all_normalized_undulation_amplitudes, bins=50)
    plt.title('Undulation Amplitude / Length')
    plt.tight_layout()
    # plt.savefig('undulation_amplitudes_normalized.png', dpi=300)
    plt.show(block=True)
    plt.close()
