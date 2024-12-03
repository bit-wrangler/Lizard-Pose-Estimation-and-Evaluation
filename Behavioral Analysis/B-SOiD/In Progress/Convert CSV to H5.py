import deeplabcut
print(deeplabcut.__version__)

# Define the paths to the config.yaml and the CSV file you want to convert
config_path = r'C:\Users\Username\Path\Bsoid\fish_run_multi\control\config.yaml' 
csv_path = r'C:\Users\Username\Path\Bsoid\fish_run_multi\control\MWA1716_chrysonotus_vedio_cropped.csv' 
output_h5_path = r'C:\Users\Username\Path\Bsoid\fish_run_multi\control\MWA1716_chrysonotus_vedio_cropped.h5' 

# Convert the CSV to HDF5
deeplabcut.convertcsv2h5(config_path)

print(f"CSV converted to H5 and saved to {output_h5_path}")
