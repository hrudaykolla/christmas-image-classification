# Christmas Images Classification

<img align="center" src="./images/Christmas_Images_Classification.png" width="750">

## Introduction


## Requirements

See requirements.txt file

## Setup

1.  Install PyTorch and other required Python libraries in a virtual environment with:

    ```
    pip install -r requirements.txt
    ```

2.  Download Data: Download data and follow the below image for hierarchy.

   <img align="center" src="./images/data_folder_setup.png" width="500">
    

## Usage

`python main.py` executes and runs the code with all the default arguments.

For changing the default arguments follow the below instructions:

1. model selection: 

    pretrained convnext_tiny model as a model: `python main.py --model convnext`

    pretrained efficientnetb4 model as a model: `python main.py --model efficientnetb4`

3. model parameters:
    
    Validation and training data split ratio, Train loader Batch size, Validation loader Batch size, epochs, learning rate, and hidden units are some of the neural net parameters that can be altered through command line.

    `python main.py --model efficientnetb4 --valtrainsplit 0.2 --trainbatchsize 4 --valbatchsize 4 --epochs 30 --learningrate 0.001 --weightdecay 0.0001`
    
    `python main.py -m efficientnetb4 -vts 0.2 -tbs 4 -vbs 4 -e 30 -lr 0.001 -wd 0.0001`


## Outputs

The output of the network is saved in Outputs folder. 
Output contains:
1. weights of model with highest validation accuracy named best_model.
2. configuration.txt file with model configurations.
3. terminal_output.txt with terminal output.
4. runs folder with logs of tensorboard.
