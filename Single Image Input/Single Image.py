import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
from torchvision import models
from PIL import Image




torch.cuda.empty_cache()

vgg = models.vgg16_bn(pretrained=True)
for param in vgg.features.parameters():
    param.requires_grad = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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




loader = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean = [0.66445047,0.55465436,0.447036], std = [0.321551,0.33547384,0.3524585])])

image = Image.open(r"C:\Users\Mark\Desktop\3rd year 1st term\ECE324\Project\iphone_photos\17.jpg")
x = loader(image)
x.unsqueeze_(0)
x = x.to(device)
print(x.shape)
#torch.Size([64, 3, 224, 224])
predictions_state, predictions_type = model(x)
_, predicted_state = torch.max(predictions_state.data, 1)
_,predicted_type = torch.max(predictions_type.data, 1)
print(predicted_state,predicted_type)