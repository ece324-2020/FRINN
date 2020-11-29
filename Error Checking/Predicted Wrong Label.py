import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path




torch.cuda.empty_cache()

vgg = models.vgg16_bn(pretrained=True)
for param in vgg.features.parameters():
    param.requires_grad = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


my_transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.ToPILImage(),
  transforms.Resize((224,224)),
  transforms.ToTensor(),
  transforms.Normalize(mean = [0.66445047,0.55465436,0.447036], std = [0.321551,0.33547384,0.3524585])
  ])
dataset =  ImageFolderWithPaths(r"C:\Users\Mark\Desktop\3rd year 1st term\ECE324\Project\Clean Data\Clean Data", transform=my_transform)

train_data, valid_data, test_data = torch.utils.data.random_split(dataset, [9600,2400,3000], generator=torch.Generator().manual_seed(0))

train_loader = DataLoader(train_data, batch_size=64,shuffle= True)
valid_loader = DataLoader(valid_data, batch_size=64,shuffle= False)
test_loader = DataLoader(test_data, batch_size=64,shuffle= False)


def relabel_state(argument):
    switcher = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 1,
        6: 1,
        7: 1,
        8: 1,
        9: 1,
        10:2,
        11:2,
        12:2,
        13:2,
        14:2}
    return switcher[argument.item()]

def relabel_type(argument):
    switcher = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 0,
        6: 1,
        7: 2,
        8: 3,
        9: 4,
        10:0,
        11:1,
        12:2,
        13:3,
        14:4}
    return switcher[argument.item()]

class MyVgg(nn.Module):
    def __init__(self,originalmodel):
        super(MyVgg,self).__init__()
        vgg = originalmodel
        # Here you get the bottleneck/feature extractor
        self.vgg_feature_extractor = nn.Sequential(*list(vgg.children())[:-1])
        self.classifier1 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 3),
            nn.Sigmoid()
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 5),
            nn.Sigmoid()
                            )

    # Set your own forward pass
    def forward(self, img, extra_info=None):

        x = self.vgg_feature_extractor(img)
        x = x.view(x.size(0), -1)
        x1 = self.classifier1(x)
        x2 = self.classifier2(x)

        return x1, x2

model = torch.load(r"C:\Users\Mark\Desktop\3rd year 1st term\ECE324\Project\Transfer Learning\Model\complete.pt")




def Wrongimage(model, specialdataloader):
    loss_fnc= torch.nn.CrossEntropyLoss()
    loss_fnc =loss_fnc.to(device)
    for inputs, labels, paths in specialdataloader:
            #get batch of data
        inputs, label = inputs.to(device), labels.to(device)
        vlabel_state = torch.zeros(len(label)).to(device)
        vlabel_type = torch.zeros(len(label)).to(device)
        for k in range(len(label)):
            vlabel_state[k] = relabel_state(label[k])
            vlabel_type[k] = relabel_type(label[k])

        #run model on validation batch
        predictions_state_v, predictions_type_v = model(inputs)

            #compute loss
        batch_valid_loss_state = loss_fnc(input=predictions_state_v.squeeze(), target=vlabel_state.long())
        batch_valid_loss_type = loss_fnc(input=predictions_type_v.squeeze(), target=vlabel_type.long())

        Overall_loss_v = batch_valid_loss_state + batch_valid_loss_type
            #evaluate
        _, predicted_state_v = torch.max(predictions_state_v.data, 1)
        _,predicted_type_v = torch.max(predictions_type_v.data, 1)

        validAcc_state = (vlabel_state == predicted_state_v).sum().item() / 64
        validAcc_type = (vlabel_type == predicted_type_v).sum().item() / 64

        for l in range(len(label)):
            if vlabel_state[l] != predicted_state_v[l]:
                print("state error",paths[l])
                print("predicted state label",predicted_state_v[l].item(),"actual state label",vlabel_state[l].item())
            if vlabel_type[l] != predicted_type_v[l]:
                print("type error", paths[l])
                print("predicted type label",predicted_type_v[l].item(),"actual type label",vlabel_type[l].item())


Wrongimage(model,test_loader)