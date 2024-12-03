import h5py

# Path to the HDF5 file
file_path = r"C:\Users\Username\Path\VAME\data-cichlid\MWA1716_chrysonotus_vedioDLC_resnet_50_ChrysonotusFeb5shuffle1_300000.h5"

# Open and inspect the file
with h5py.File(file_path, "r") as hdf:
    # Print all top-level keys (groups or datasets)
    print("Keys in the file:", list(hdf.keys()))

    # Recursively explore all groups and datasets
    def explore_h5(name, obj):
        print(f"{name}: {obj}")

    hdf.visititems(explore_h5)
