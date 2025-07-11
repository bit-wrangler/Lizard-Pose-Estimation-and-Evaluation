# Imports
import os
import shutil

# Collect Groups
def organize_gif_files(base_dir):
    gif_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".gif"):
                gif_files.append(os.path.join(root, file))
    for gif_file in gif_files:
        file_name = os.path.basename(gif_file)
        parts = file_name.split('_')
        group = '_'.join(parts[:2]) 
        if not group.startswith("group_"):
            print(f"Skipping file without group information: {file_name}")
            continue
        group_folder = os.path.join(base_dir, group)
        os.makedirs(group_folder, exist_ok=True)
        counter = 0
        dest_file_path = os.path.join(group_folder, f"{group}_{counter}.gif")
        while os.path.exists(dest_file_path):
            counter += 1
            dest_file_path = os.path.join(group_folder, f"{group}_{counter}.gif")
        shutil.copy(gif_file, dest_file_path)
        print(f"Copied {gif_file} to {dest_file_path}")

# Run Function
base_directory = r"C:\Users\Username\Path\to\File\Bsoid\control\mp4s"
organize_gif_files(base_directory)