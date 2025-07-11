# Imports
import pandas as pd

# Give the Current H5 File and Desired CSV File Name
h5_file = r'C:\Users\Username\Path\Bsoid\lizard_run\control\0566_1DLC_Resnet50_20_videos_0603Oct10shuffle1_snapshot_200.h5'
csv_file = r'C:\Users\Username\Path\Bsoid\lizard_run\control\0566_1DLC_Resnet50_20_videos_0603Oct10shuffle1_snapshot_200.csv'

# Convert H5 to CSV
data = pd.read_hdf(h5_file)
data.to_csv(csv_file)