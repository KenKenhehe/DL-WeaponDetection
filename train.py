import torch
from utils_dataset import WeaponDetectionDataset
from torch.utils.data import random_split
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
import torch.nn as nn
from torchvision.models import resnet, vgg16

import os

from util_dls import train, test, MLPNet, CNN
from res_net_arch import ResNet, Block
import cv2
from tqdm import tqdm

transform = transforms.ToTensor()

batch_size = 100

data_file_list = []
data_folder_path = "preprocessed_img"

split_perccentage = 0.75

LABELS = {"HasWeapon": 0, "NoWeapon": 1}

num_labels = len(LABELS)
print(num_labels)
# preprocess and load data from disk
for root, dirs, files in os.walk(data_folder_path):
    for file in tqdm(files, desc="loading data"):
        label = root.split("\\")[-1]
        data = {}
        if file.lower().endswith(".jpg") or file.lower().endswith(".png"):
            data["image"] = cv2.imread(os.path.join(root, file))#, cv2.IMREAD_GRAYSCALE
            data["image"] = cv2.normalize(data["image"], data["image"], 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            data["label"] = LABELS[label]
            data_file_list.append(data)

animal_dataset = WeaponDetectionDataset(data_file_list, transform=transform)

total_size = len(animal_dataset)
train_size = int(split_perccentage * total_size)
test_size = total_size - train_size

train_dataset, test_dataset = random_split(animal_dataset, [train_size, test_size])

train_total_size = len(train_dataset)
train_size = int(split_perccentage * train_total_size)
val_size = train_total_size - train_size

train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

print(f"total size: {total_size}")
print(f"train size: {len(train_dataset)}")
print(f"test size: {len(test_dataset)}")
print(f"validation size: {len(val_dataset)}")

train_loader = dataloader.DataLoader(train_dataset, batch_size=batch_size, 
                                    num_workers=0, shuffle=True)
validation_loader = dataloader.DataLoader(val_dataset, batch_size=batch_size, 
                                    num_workers=0, shuffle=True)
test_loader = dataloader.DataLoader(test_dataset, batch_size=batch_size, 
                                    num_workers=0, shuffle=True)

data, label = train_dataset[0]

dim, size_x, size_y = data.shape
print(f"Training data shape = {data.shape}")

#model = MLPNet(size_x, size_y)
#model = CNN(size_x, size_y)
#model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=5)

# Load pre-trained VGG model
model = vgg16(pretrained=True)

# Modify the classifier for your specific case (assuming 2 classes)
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 2)

#best parameter: SGD with lr: 0.0005
train(model=model, train_loader=train_loader, validation_loader= validation_loader, 
      criterion=nn.CrossEntropyLoss(), optimizer=torch.optim.SGD(model.parameters(), lr=0.0005), n_epochs=150)

model.load_state_dict(torch.load('weapon_detetion_model_VGG.pt'))
print("testing on current best")
test(model=model, test_loader=test_loader,
      criterion=nn.CrossEntropyLoss())