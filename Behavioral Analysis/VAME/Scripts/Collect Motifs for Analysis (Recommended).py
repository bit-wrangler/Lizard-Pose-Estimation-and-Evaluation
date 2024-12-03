# Imports
import os
import shutil

# Collect the Motif Examples from All Subdirectories
def organize_mp4_files(base_dir):
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".mp4"):
                parts = file.split('-')
                motif = [part for part in parts if part.startswith('motif_')]
                if motif:
                    motif_folder_name = motif[0]
                    motif_folder_path = os.path.join(base_dir, motif_folder_name)
                    os.makedirs(motif_folder_path, exist_ok=True)
                    src_file_path = os.path.join(root, file)
                    dest_file_path = os.path.join(motif_folder_path, file)
                    shutil.move(src_file_path, dest_file_path)
                    print(f"Moved {file} to {motif_folder_path}")

# Run Function
base_directory = r"C:/Users/Username/VAME/data/lizard project-Nov27-2024/results"
organize_mp4_files(base_directory)

# Source: ChatGPT