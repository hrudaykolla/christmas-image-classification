import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def get_device():
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device

def get_output_folder(base_path):
    # Look for existing output folders and find the highest numbered one
    existing_folders = [folder for folder in os.listdir(base_path) if folder.startswith("output")]
    max_num = 0
    for folder in existing_folders:
        try:
            num = int(folder.split("_")[-1])
            if num > max_num:
                max_num = num
        except ValueError:
            pass

    # Increment the number and create the new output folder
    new_folder_name = os.path.join(base_path, f"output_{max_num + 1}")
    os.makedirs(new_folder_name, exist_ok=True)
    return new_folder_name