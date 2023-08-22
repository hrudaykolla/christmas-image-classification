import torch
from torch.utils.data import DataLoader
from eval_utils import TestSet, evaluate
from model import Network

# dataset location
path = 'dataset'

model = Network().eval()
model.load_state_dict(torch.load('model'))

loader = DataLoader(TestSet(path), batch_size=1)

print(evaluate(model, loader))