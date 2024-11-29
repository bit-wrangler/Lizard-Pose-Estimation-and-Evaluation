# Imports
import vame
from pathlib import Path

# Set Up Variables
working_directory = "C:/Users/kaiwi/OneDrive/Documents/VAME/data/"
working_directory_path = Path("C:/Users/kaiwi/OneDrive/Documents/VAME/data/")
project = "lizard project"
videos = []
poses_estimations = []

# Initialize Project
config = 'C:/Users/kaiwi/OneDrive/Documents/VAME/data/lizard project-Nov27-2024/config.yaml'

# Generative Reconstruction Decoder -- Failed here, hmm not for for centers mode
vame.generative_model(config, mode="centers", parametrization='kmeans')

# Output Video -- Failed here, missing parametrization -- Failed here, param should be hmm
# -- Failed here, pose_ref_index lowered by 1??
# -- Failed here, changed subtract background to false
# -- Failed here, crop_size must be square
vame.gif(config, pose_ref_index=[0,4], subtract_background=False, start=None, parametrization='hmm',
         length=100, max_lag=30, label='motif', file_format='.mp4', crop_size=(200,200))

# Source: VAME Documentation at https://ethoml.github.io/VAME/docs/getting_started/running
# Source: ChatGPT