from scipy.signal import find_peaks
import numpy as np
import h5py
import glob
import os
import matplotlib.pyplot as plt
import tqdm
from utils.landmarks import *

PROCESSED_DATA_DIR = 'data/cdl-projects/test1-haag-2025-05-21/processed'

EVENTS_H5_PATH = os.path.join(PROCESSED_DATA_DIR, 'stride_events.h5')

EVENT_THRESHOLD = 0.5

SCALE = 1.75 # px/mm

ANNOTATION_SET = [
    "left ankle placed",
    "right ankle placed",
    "left wrist placed",
    "right wrist placed",
]

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

def get_stride_length(
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
    distances = distances / SCALE
    confidences = limb_position_data[event_peaks[1:], 2] * limb_position_data[event_peaks[:-1], 2]
    valid_mask = confidences > min_confidence_pos
    valid_mask = valid_mask & (distances > 0.01) & (distances < body_length * 6.0)
    return distances[valid_mask].max() if np.any(valid_mask) else None
    

if __name__ == "__main__":
    kinematic_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, '*_velocity.h5'))
    video_kinematic_map = {}
    for file in kinematic_files:
        video_id = os.path.basename(file).split('_velocity')[0]
        video_kinematic_map[video_id] = file

    videos_with_results = 0
    all_rel_stride_lengths = []

    with  h5py.File(EVENTS_H5_PATH, 'r') as f:
        for video_id in tqdm.tqdm(f.keys()):
            if video_id not in video_kinematic_map:
                continue
            # print(f"Processing {video_id}...")

            kinematic_file = video_kinematic_map[video_id]
            with h5py.File(kinematic_file, 'r') as kinematic_f:
                position_data = kinematic_f['position_x_y_conf'][:]

            lizard_length = get_lizard_base_length(position_data)
            if np.isnan(lizard_length):
                # print("Warning: Could not estimate lizard length. Skipping.")
                continue
            # print(f"Lizard length: {lizard_length}")

            events = f[video_id][:]
            left_ankle_stride_length = get_stride_length(
                position_data[:, LEFT_ANKLE, :],
                events[:, ANNOTATION_SET.index("left ankle placed")],
                lizard_length
            )
            right_ankle_stride_length = get_stride_length(
                position_data[:, RIGHT_ANKLE, :],
                events[:, ANNOTATION_SET.index("right ankle placed")],
                lizard_length
            )
            left_wrist_stride_length = get_stride_length(
                position_data[:, LEFT_WRIST, :],
                events[:, ANNOTATION_SET.index("left wrist placed")],
                lizard_length
            )
            right_wrist_stride_length = get_stride_length(
                position_data[:, RIGHT_WRIST, :],
                events[:, ANNOTATION_SET.index("right wrist placed")],
                lizard_length
            )
            # print(f"Left Ankle: {left_ankle_stride_length}")
            # print(f"Right Ankle: {right_ankle_stride_length}")
            # print(f"Left Wrist: {left_wrist_stride_length}")
            # print(f"Right Wrist: {right_wrist_stride_length}")

            any_valid = (
                left_ankle_stride_length is not None or
                right_ankle_stride_length is not None or
                left_wrist_stride_length is not None or
                right_wrist_stride_length is not None
            )

            if any_valid:
                videos_with_results += 1

            max_leg_stride = -1
            if left_ankle_stride_length is not None:
                max_leg_stride = max(max_leg_stride, left_ankle_stride_length)
            if right_ankle_stride_length is not None:
                max_leg_stride = max(max_leg_stride, right_ankle_stride_length)
            
            if max_leg_stride > 0:
                all_rel_stride_lengths.append(max_leg_stride / lizard_length)

            # if left_ankle_stride_length is not None:
            #     all_rel_stride_lengths.append(left_ankle_stride_length / lizard_length)
            # if right_ankle_stride_length is not None:
            #     all_rel_stride_lengths.append(right_ankle_stride_length / lizard_length)
            # if left_wrist_stride_length is not None:
            #     all_rel_stride_lengths.append(left_wrist_stride_length / lizard_length)
            # if right_wrist_stride_length is not None:
            #     all_rel_stride_lengths.append(right_wrist_stride_length / lizard_length)

        print(f"Processed {videos_with_results} videos with valid stride length estimates.")
        print(f"Average relative stride length: {np.mean(all_rel_stride_lengths)}")

        plt.hist(all_rel_stride_lengths, bins=50)
        plt.title('Relative Rear Leg Stride Length')
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig('relative_rear_leg_stride_lengths.png', dpi=300)
        plt.close()
    