import util_dls
from util_dls import predict
import torch.nn as nn
from torchvision.models import vgg16
import torch

# Load pre-trained VGG model
model = vgg16(pretrained=True)

# Modify the classifier for your specific case (assuming 2 classes)
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 2)

model.load_state_dict(torch.load('weapon_detetion_model_VGG.pt'))

if __name__ == "__main__":
    print("Image contains weapon" if predict(model, f"ForInference/HasWeapon/130.jpg") == 0 else "Image doesn't contain weapon")
    print("Image contains weapon" if predict(model, f"ForInference/HasWeapon/141.jpg")== 0 else "Image doesn't contain weapon")
    print("Image contains weapon" if predict(model, f"ForInference/HasWeapon/176.jpg")== 0 else "Image doesn't contain weapon")
    print("Image contains weapon" if predict(model, f"ForInference/HasWeapon/192.jpg")== 0 else "Image doesn't contain weapon")
    
    print("Image contains weapon" if predict(model, f"ForInference/NoWeapon/60.jpg")== 0 else "Image doesn't contain weapon")
    print("Image contains weapon" if predict(model, f"ForInference/NoWeapon/77.jpg")== 0 else "Image doesn't contain weapon")
    print("Image contains weapon" if predict(model, f"ForInference/NoWeapon/83.jpg")== 0 else "Image doesn't contain weapon")
    print("Image contains weapon" if predict(model, f"ForInference/NoWeapon/101.jpg")== 0 else "Image doesn't contain weapon")