# Imports
import os
import shutil

# Collect the Group Examples from All Subdirectories
def organize_gif_files(base_dir):
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".gif"):
                parts = file.split('_')
                group = '_'.join(parts[:2]) 
                group_folder_path = os.path.join(base_dir, group)
                os.makedirs(group_folder_path, exist_ok=True)
                src_file_path = os.path.join(root, file)
                dest_file_path = os.path.join(group_folder_path, file)
                shutil.move(src_file_path, dest_file_path)
                print(f"Moved {file} to {group_folder_path}")

# Run Function
base_directory = r"C:\Users\Username\Path\to\Directory"
organize_gif_files(base_directory)

# Source: ChatGPT