# Imports
import os
import glob
import pandas as pd

# Convert H5 Files to CSV
def convert_h5_to_csv(directory_path):
    h5_files = glob.glob(os.path.join(directory_path, '*.h5'))
    
    for h5_file in h5_files:
        base_name = os.path.splitext(os.path.basename(h5_file))[0]
        csv_file = os.path.join(directory_path, f'{base_name}.csv')
        
        try:
            data = pd.read_hdf(h5_file)
            data.to_csv(csv_file, index=False)
            print(f"Converted {h5_file} to {csv_file}")
        except Exception as e:
            print(f"Error converting {h5_file}: {e}")

# Call Function
directory_path = 'C:/Users/kaiwi/OneDrive/Documents/VAME/data'
convert_h5_to_csv(directory_path)

# Source: Chat GPT + Previous Script