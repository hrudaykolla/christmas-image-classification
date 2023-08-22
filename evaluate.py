import os
import sys
import torch
from torch.utils.data import DataLoader
from eval_utils import TestSet, evaluate
from models.convnext_model import convnext_model
from models.efficientnetb4_model import efficientnetb4_model

from func import text_to_dict

# dataset location
path = './data/test'
# Output path location
ouput_path = './outputs/output_1'

terminal_output_file = os.path.join(ouput_path, "test_terminal_output.txt")
sys.stdout = open(terminal_output_file, "w")

config_txt_path = os.path.join(ouput_path, 'configuration.txt')
config = text_to_dict(config_txt_path)

if config["model"] == "convnext":
    model = convnext_model()
elif config["model"] == "efficientnetb4":
    model = efficientnetb4_model()

model_path = os.path.join(ouput_path, 'best_model')
model.load_state_dict(torch.load(model_path))

loader = DataLoader(TestSet(path), batch_size=1)

accuracy = evaluate(model, loader, ouput_path)