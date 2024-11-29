# Imports
import vame
from pathlib import Path

# Set Up Variables
working_directory = "C:/Users/kaiwi/OneDrive/Documents/VAME/data/"
working_directory_path = Path("C:/Users/kaiwi/OneDrive/Documents/VAME/data/")
project = "lizard project"
videos = []
poses_estimations = []

# Go Through Working Directory and Add All Files
for file_path in working_directory_path.iterdir():
    if file_path.is_file():
        if file_path.suffix == '.mp4':
            videos.append("C:/Users/kaiwi/OneDrive/Documents/VAME/data/"+ file_path.name)
        elif file_path.suffix == '.csv':
            poses_estimations.append("C:/Users/kaiwi/OneDrive/Documents/VAME/data/" + file_path.name)

# Initialize Project
config = vame.init_new_project(
    project = project,
    videos = videos,
    poses_estimations=poses_estimations,
    working_directory=working_directory_path,
    videotype='.mp4'
)

# Transform DLC CSVs to Numpy Arrays
vame.egocentric_alignment(config, pose_ref_index=[0,5])

# Create Training Dataset for Videos with 6 Keypoints
vame.create_trainset(config, pose_ref_index=[0,5])

# Train the Model
vame.train_model(config)

# Evaluate the Model
vame.evaluate_model(config)

# Pose Segmentation
vame.pose_segmentation(config)

# Create Motif Videos -- Failed Here
vame.motif_videos(config, videoType='.mp4', parametrization='hmm')

# Run Community Detection
vame.community(config, parametrization='hmm', cut_tree=2, cohort=False)

# UMAP Visualization
fig = vame.visualization(config, label='community')

# Generative Reconstruction Decoder
vame.generative_model(config, mode="centers")

# Output Video
vame.gif(config, pose_ref_index=[0,5], subtract_background=True, start=None,
         length=360, max_lag=30, label='community', file_format='.mp4', crop_size=(300,300))

# Source: VAME Documentation at https://ethoml.github.io/VAME/docs/getting_started/running
# Source: ChatGPT