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

def text_to_dict(path_to_text):
    config = {}

    with open(path_to_text, "r") as file:
        lines = file.readlines()

    for line in lines:
        key, value = line.strip().split(":")
        key = key.strip()
        value = value.strip()

        # Handle special cases where keys have spaces or mixed case
        if key == "val train split":
            key = "val_train_split"
        elif key.lower() == "epochs":
            key = "epochs"
        elif key.lower() == "learning rate":
            key = "learning_rate"
        elif key.lower() == "train batch size":
            key = "train_batch_size"
        elif key.lower() == "validation batch size":
            key = "validation_batch_size"
        elif key.lower() == "weight decay":
            key = "weight_decay"
        
        # Convert value to appropriate data type if needed
        if key in ["val_train_split", "epochs", "learning_rate", "train_batch_size", "validation_batch_size", "weight_decay"]:
            if "." in value:
                value = float(value)
            else:
                value = int(value)

        config[key] = value

    return config