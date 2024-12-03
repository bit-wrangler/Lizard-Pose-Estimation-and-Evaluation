import os
import shutil

# Define the source folder and target folder
source_folder = r"C:\Users\Username\Path\Bsoid\fish_run\control\pngs\all_frames"
target_folder = r"C:\Users\Username\Path\Bsoid\fish_run\control\pngs\MWA1716_chrysonotus_vedio"

# Create the target folder if it doesn't exist
os.makedirs(target_folder, exist_ok=True)

# Get a list of files in the source folder
files = sorted(os.listdir(source_folder))  # Sort to ensure consistent ordering

# Copy the first 9,900 files
for i, file_name in enumerate(files[:9900]):
    source_file = os.path.join(source_folder, file_name)
    target_file = os.path.join(target_folder, file_name)
    shutil.copy2(source_file, target_file)

print("Copying completed.")

