import cv2
import pandas as pd
import random
import os

# Parameters
video_path = r"C:\Users\Username\Path\VAME\data-cichlid\MWA1716_chrysonotus_vedio.MP4"
csv_path = r"C:\Users\Username\Path\VAME\data-cichlid\MWA1716_chrysonotus_vedio.csv"
output_video = r"C:\Users\Username\Path\VAME\data-cichlid\MWA1716_chrysonotus_vedio_cropped.MP4"
output_csv = r"C:\Users\Username\Path\VAME\data-cichlid\MWA1716_chrysonotus_vedio_cropped.csv"
clip_duration = 4 * 60
fps = 50 
frames_to_extract = clip_duration * fps

# Step 1: Load the video and calculate total frames
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_rate = cap.get(cv2.CAP_PROP_FPS)

# Ensure FPS consistency
if frame_rate != fps:
    print("Warning: Frame rate mismatch. Adjust 'fps' parameter to match your video.")

# Step 2: Choose a random start frame
max_start_frame = total_frames - frames_to_extract
if max_start_frame <= 0:
    raise ValueError("Video too short for a 4-minute clip.")
start_frame = random.randint(0, max_start_frame)

# Step 3: Clip the video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

frames_written = 0
while frames_written < frames_to_extract:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    frames_written += 1

cap.release()
out.release()

# Step 4: Extract corresponding rows from the CSV while keeping the first three rows
data = pd.read_csv(csv_path, header=None)

# Separate metadata (first 3 rows) and actual data
metadata = data.iloc[:3]
data_rows = data.iloc[3:]  # Actual data starts after the metadata

# Ensure the data rows match the video frame count
if len(data_rows) != total_frames:
    raise ValueError("The CSV data rows (excluding metadata) do not match the video frame count!")

# Extract the rows corresponding to the selected video frames
start_row = start_frame
end_row = start_frame + frames_to_extract
clipped_data_rows = data_rows.iloc[start_row:end_row]

# Combine metadata with the clipped data
clipped_csv = pd.concat([metadata, clipped_data_rows])

# Save the clipped CSV
clipped_csv.to_csv(output_csv, index=False, header=False)

print(f"Clipped video saved to {output_video}")
print(f"Clipped CSV saved to {output_csv}")

# Source: ChatGPT