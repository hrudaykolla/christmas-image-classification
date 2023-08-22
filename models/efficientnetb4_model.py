import torch
import torch.nn as nn
import torchvision

class efficientnetb4_model(nn.Module):
    
    def __init__(self):
        super().__init__()

        # Load the pretrained EfficientNet model
        self.model = torchvision.models.efficientnet_b4(pretrained=True)

        # Freeze all the layers except the last one
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier[-1].requires_grad = True

        # Replace the last layer with a new one for 8 classes
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, 8)   
        
    def forward(self, x):
        
        #############################
        # Implement the forward pass
        #############################
        
        return self.model(x)
        pass
    
    def save_model(self):
        
        #############################
        # Saving the model's weitghts
        # Upload 'model' as part of
        # your submission
        # Do not modify this function
        #############################
        
        torch.save(self.state_dict(), 'model')

