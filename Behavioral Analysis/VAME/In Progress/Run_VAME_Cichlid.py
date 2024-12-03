# Imports
import vame
from pathlib import Path

# Set Up Variables
working_directory = "C:/Users/Username/Path/VAME/data-cichlid/"
working_directory_path = Path("C:/Users/Username/Path/VAME/data-cichlid/")
project = "cichlid project"
videos = ["C:/Users/Username/Path/VAME/data-cichlid/MWA1716_chrysonotus_vedio.mp4",
          "C:/Users/Username/Path/VAME/data-cichlid/MWA1717_chrysonotus_vedio.mp4"]
poses_estimations = ["C:/Users/Username/Path/VAME/data-cichlid/MWA1716_chrysonotus_vedio.csv",
                     "C:/Users/Username/Path/VAME/data-cichlid/MWA1717_chrysonotus_vedio.csv"]

# Initialize Project
#'''
config = vame.init_new_project(
    project = project,
    videos = videos,
    poses_estimations=poses_estimations,
    working_directory=working_directory_path,
    videotype='.mp4'
)
#'''

# Run an Existing Project
# config = 'C:/Users/Username/Path/VAME/data-cichlid/cichlid project-Dec1-2024/config.yaml'

# Transform DLC CSVs to Numpy Arrays
# vame.pose_to_numpy(config)

# Egocentrically Align
vame.egocentric_alignment(config, pose_ref_index=[0,2])

# Create Training Dataset for Videos with 3 Keypoints
vame.create_trainset(config)

# Train the Model --err
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
vame.gif(config, pose_ref_index=[0,2], subtract_background=False, start=None, parametrization='hmm',
         length=100, max_lag=30, label='motif', file_format='.mp4', crop_size=(200,200))

# Source: VAME Documentation at https://ethoml.github.io/VAME/docs/getting_started/running
# Source: ChatGPT