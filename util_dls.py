import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torch_data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2

from tqdm import tqdm


class MLPNet(nn.Module):
    def __init__(self, input_x_size: int, input_y_size: int):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(input_x_size * input_y_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 2)

        self.dropout = nn.Dropout(0.1)
        self.input_x_size = input_x_size
        self.input_y_size = input_y_size

    def forward(self, x: torch.Tensor):
        x = x.view(-1, self.input_x_size * self.input_y_size)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def __str__(self) -> str:
       return "MLP"
        

class CNN(nn.Module):
    def __init__(self, input_x_size: int, input_y_size: int):
        super().__init__()
        self.input_x_size = input_x_size
        self.input_y_size = input_y_size

        self.conv1 = nn.Conv2d(1, 32, 5)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 2)
        self.bn4 = nn.BatchNorm2d(256)

        x = torch.randn(input_x_size, input_y_size).view(-1, 1, input_x_size, input_y_size)
        self._to_linear = None
        self.conv(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def conv(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), (2, 2))
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), (2, 2))
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), (2, 2))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim = 1)

    def __str__(self) -> str:
        return "CNN"

def train(model: torch.nn.Module, 
          train_loader:torch_data.DataLoader, 
          validation_loader: torch_data.DataLoader,
          criterion = None, 
          optimizer = None, 
          n_epochs: int = 10 ):
    if(torch.cuda.is_available()):
        device = "cuda"
        print("CUDA Avaliable, using GPU to train")
    else:
        device = "cpu"
    model = model.to(device)

    train_loss = 0.0

    validation_loss_min = np.Inf
    validation_loss = 0.0
    for epoch in range(n_epochs):
        model.train()
        for data, target in tqdm(train_loader, desc= f"Epoch {epoch + 1}/{n_epochs}"):
            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = criterion(output, target)
            if torch.isnan(loss):
                print(f"input data is {data}")
                print(f"output nan is {output}")
                print("Loss is NaN. Stopping training.")
                return
            loss.backward()
            optimizer.step()
            
            if epoch > 75:
             for g in optimizer.param_groups:
                g['lr'] = 0.0001

            train_loss += loss.item() * data.size(0)
           

        model.eval()
        for data, target in tqdm(validation_loader, desc= f"Validating Epoch {epoch + 1}/{n_epochs} "):
            output = model(data.to(device))
            output = output.to(device)
            target = target.to(device)

            #output = model(data)

            loss = criterion(output, target)
            
            validation_loss += loss.item() * data.size(0)

        train_loss = train_loss / len(train_loader.dataset) 
        validation_loss = validation_loss / len(validation_loader.dataset)

        if validation_loss < validation_loss_min:
            print(f"Validation loss decreased ({validation_loss_min} --> {validation_loss}).  Saving model ...")
            
            torch.save(model.state_dict(), f"weapon_detection_model_{type(model).__name__}.pt")
            validation_loss_min = validation_loss

        print(f"Epoch: {epoch+1} \tTraining Loss: {train_loss}. Validation Loss: {validation_loss}")

def test(model: torch.nn.Module, 
        test_loader:torch_data.DataLoader, 
        criterion = None):
    if(torch.cuda.is_available()):
        device = "cuda"
    else:
        device = "cpu"
    num_class = 2

    model = model.to(device)
    model.eval()

    test_loss = 0.0
    class_correct = list(0. for i in range(num_class))
    class_total = list(0. for i in range(num_class))

    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        loss = criterion(output, target)

        # update test loss 
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)

        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))

        # calculate test accuracy for each object class
        for i in range(len(target.data)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    for i in range(num_class):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))

    # calculate and print avg test loss
    test_loss = test_loss/len(test_loader.dataset)

    #accuracy
    # test_acc = class_correct[label] /class_total[label]
    # print(f"total: {class_total[label]}, correct: {class_correct[label]}")
    # print('Test acc: {:.6f}'.format(test_acc))
    print('Test Loss: {:.6f}'.format(test_loss))
    
def predict(model: torch.nn.Module, img_path: str):
    #output = model(img)
    if(torch.cuda.is_available()):
        device = "cuda"
    else:
        device = "cpu"
    model = model.to(device)
    model.eval()
    transform = transforms.ToTensor()
    IMG_SIZE = 100
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    transformed_image = transform(img).to(device)
    
    output = model(transformed_image.unsqueeze(0))
    output = torch.squeeze(output).argmax().item()
    return output