import os
import sys
import time

import torch
import torch.nn as nn

from torch.utils.data import random_split
from torch.utils.data import DataLoader
from multiprocessing import freeze_support

from models.convnext_model import convnext_model
from models.efficientnetb4_model import efficientnetb4_model
from data import ChristmasImages
from func import get_device, get_output_folder

import argparse

def main():
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(os.path.join(output_path,'runs/'))
    #Extracting data
    data_path = "./data/train_val"
    dataset = ChristmasImages(data_path, training = True)
    print(f"No of Images available in data are: {len(dataset)}")
    #split the available data to validation and training
    val_data_size = int(args.valtrainsplit * len(dataset))
    train_data_size = (len(dataset) - val_data_size)

    #randomly split the data
    train_data, val_data = random_split(dataset,[train_data_size,val_data_size])
    print(f"Number of Images used for training: {len(train_data)}")
    print(f"Number of Images used for Validation: {len(val_data)}")

    #data loader for training and validation
    train_dl = DataLoader(train_data, batch_size = args.trainbatchsize, shuffle = False, num_workers = 4, pin_memory = True)
    val_dl = DataLoader(val_data, batch_size = args.valbatchsize, shuffle = False, num_workers = 4, pin_memory = True)

    if args.model == "convnext":
        model = convnext_model()
    elif args.model == "efficientnetb4":
        model = efficientnetb4_model()

    device = get_device()

    optimizer = torch.optim.AdamW(model.parameters(), lr = args.learningrate, weight_decay = args.weightdecay)

    loss_function = nn.CrossEntropyLoss()

    best_val_acc = 0
    for epoch in range(args.epochs):
        total_train_loss = 0
        train_correct = 0
        train_total = 0
        
        time_start = time.time()
        model.to(device)
        # Training step
        model.train()
        for data, target in train_dl:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        train_acc = train_correct / train_total
        train_loss = total_train_loss / len(train_dl)

        # Evaluation step
        with torch.no_grad():
            model.eval()
            total_val_loss = 0
            val_correct = 0
            val_total = 0
            for data, target in val_dl:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                val_loss = loss_function(output, target)
                total_val_loss += val_loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
            val_acc = val_correct / val_total
            val_loss = total_val_loss / len(val_dl)
            
            if val_acc>best_val_acc:
                model.to('cpu')
                best_val_acc = val_acc
                best_weights = model.state_dict()
                path = os.path.join(output_path, 'best_model')
                torch.save(best_weights, path)
        
        writer.add_scalars("loss_graphs",{"Loss/train" : train_loss,"Loss/val" : val_loss}, epoch)
        writer.add_scalars("accuracy_graphs",{"accuracy/train" : train_acc, "accuracy/val" : val_acc}, epoch)

        time_end = time.time()
        time_elapsed = time_end - time_start
        print("Time_elapsed for Epoch [{}] : [{:.2f}] s".format(epoch+1,time_elapsed))
        print(f'Training Loss: {train_loss:.4f} | Train Acc: {100*train_acc:.4f}')
        print(f'Validation Loss: {val_loss:.4f} | Validation Acc: {100*val_acc:.4f} | Best Validation Acc: {100*best_val_acc:.4f}')

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #model
    parser.add_argument("-m",
                        "--model", 
                        type = str,
                        default= 'efficientnetb4',
                        choices = ['convnext','efficientnetb4'],
                        help="choice of model",
                        )
    parser.add_argument("-vts",
                        "--valtrainsplit", 
                        type=float,
                        default=0.15,
                        help="validation and train data split",
                        )
    parser.add_argument("-tbs",
                        "--trainbatchsize", 
                        type=int,
                        default= 4,
                        help="batch size for train dataloader",
                        )
    parser.add_argument("-vbs",
                        "--valbatchsize", 
                        type=int,
                        default= 4,
                        help="batch size for validation dataloader",
                        )
    parser.add_argument("-e",
                        "--epochs", 
                        type=int,
                        default= 5,
                        help="no of epochs",
                        )
    parser.add_argument("-lr",
                        "--learningrate", 
                        type=float,
                        default= 0.001,
                        help="learning rate",
                        )
    parser.add_argument("-wd",
                        "--weightdecay", 
                        type=float,
                        default= 0.0001,
                        help="weight decay",
                        )
    args = parser.parse_args()

    base_path = "./outputs/"
    os.makedirs(base_path, exist_ok=True)
    output_path = get_output_folder(base_path)
    print(output_path)
    config_file_path = os.path.join(output_path, "configuration.txt")
        
    with open(config_file_path, "w") as file:
        file.write("model: {}\n".format(args.model))
        file.write("val train split: {}\n".format(args.valtrainsplit))
        file.write("Epochs: {}\n".format(args.epochs))
        file.write("Learning Rate: {}\n".format(args.learningrate))
        file.write("train batch size: {}\n".format(args.trainbatchsize))
        file.write("validation batch size: {}\n".format(args.valbatchsize))
        file.write("weight decay: {}\n".format(args.weightdecay))

    terminal_output_file = os.path.join(output_path, "terminal_output.txt")
    sys.stdout = open(terminal_output_file, "w")
    main()