import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

class ChristmasImages(Dataset):
    
    def __init__(self, path, training=True):
        super().__init__()
        self.training = training
        self.path = path
        
        # Mean and std deviation of Image net images
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        # If training == True, path contains subfolders
        # containing images of the corresponding classes
        if self.training == True:
           
            train_transforms = transforms.Compose([
                transforms.Resize((236,236)),
                transforms.CenterCrop(224),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
            ])

            #load the train dat
            self.dataset = ImageFolder(root = self.path, transform = train_transforms)
            
        # If training == False, path directly contains
        # the test images for testing the classifier
        
        if self.training == False:
          
            list_dir=[int(file.split(".")[0]) for file in os.listdir(self.path)]
            list_dir.sort()

            test_images= []

            test_transforms = transforms.Compose([
                transforms.Resize((236,236)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

            for fname in list_dir:  
                filename = self.path + '/' + str(fname)+".png"
                im=Image.open(filename).convert('RGB')
                im_tensor = test_transforms(im)
                test_images.append(im_tensor)

            self.dataset = torch.stack(test_images)
            

    def __getitem__(self, index):
        if self.training == True:
            img,label = self.dataset[index]
            return img, label
        
        # If self.training == False, output (image, )
        # where image will be used as input for your model
        
        if self.training == False:
            img = self.dataset[index]
            return (img,)
        
        raise NotImplementedError  
        
   
    def __len__(self):
        return len(self.dataset)
