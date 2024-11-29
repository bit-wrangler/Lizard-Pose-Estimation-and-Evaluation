# Imports
import vame
from pathlib import Path

# Set Up Variables
working_directory = "C:/Users/Username/Path/To/Files/VAME/data/"
working_directory_path = Path("C:/Users/Username/Path/To/Files/VAME/data/")
project = "lizard project"
videos = []
poses_estimations = []

# Go Through Working Directory and Add All Files
for file_path in working_directory_path.iterdir():
    if file_path.is_file():
        if file_path.suffix == '.mp4':
            videos.append("C:/Users/Username/Path/To/Files/VAME/data/"+ file_path.name)
        elif file_path.suffix == '.csv':
            poses_estimations.append("C:/Users/Username/Path/To/Files/VAME/data/" + file_path.name)

# Initialize Project
config = vame.init_new_project(
    project = project,
    videos = videos,
    poses_estimations=poses_estimations,
    working_directory=working_directory_path,
    videotype='.mp4'
)

# Run an Existing Project
# config = 'C:/Users/Username/Path/To/Files/VAME/data/lizard project-Day-2024/config.yaml'

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

# Create Motif Videos
vame.motif_videos(config, videoType='.mp4', parametrization='hmm')

# Run Community Detection 
vame.community(config, parametrization='hmm', cut_tree=None, cohort=True)

# UMAP Visualization
fig = vame.visualization(config, label=None, parametrization='hmm')

# Generative Reconstruction Decoder
vame.generative_model(config, mode="centers", parametrization='kmeans')

# Output Video
vame.gif(config, pose_ref_index=[0,4], subtract_background=False, start=None, parametrization='hmm',
         length=100, max_lag=30, label='motif', file_format='.mp4', crop_size=(200,200))

# Source: VAME Documentation at https://ethoml.github.io/VAME/docs/getting_started/running
# Source: ChatGPT