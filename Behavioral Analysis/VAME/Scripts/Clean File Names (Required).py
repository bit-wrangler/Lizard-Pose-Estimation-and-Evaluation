# Imports
import os
from pathlib import Path

# Working Path
working_directory = Path("C:/Users/kaiwi/OneDrive/Documents/VAME/data/")

# Remove Strings that Cause Mismatch of CSV and MP4 Files
for file_path in working_directory.glob("*.mp4"):
    if "_p60_labeled" in file_path.name:
        new_name = file_path.name.replace("_p60_labeled", "")
        new_file_path = file_path.with_name(new_name)
        file_path.rename(new_file_path)

# Source: ChatGPT