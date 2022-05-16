from pickletools import optimize
import torch
import torchvision
import torchvision.transforms as transforms
import os
import cv2
from dataset import Dataset
from torch.utils.data import DataLoader
from time import time
import torch.nn as nn
import torch.nn.functional as F
from visdom import Visdom
import numpy as np

# 设置visdom
# viz = Visdom()
# step_list = [0]
# win = viz.line(X=np.array([0]), Y=np.array([1.0]), opts=dict(title='loss'))

# class GarmentClassifier(nn.Module):
#     def __init__(self):
#         super(GarmentClassifier, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 4 * 4, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 4 * 4)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# model = GarmentClassifier().cuda()
model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=False).cuda()
    
BASE_DIR = 'D:/CV Course/FashionMNIST/data/'
impath = os.listdir(BASE_DIR + 'train')

train_ds = Dataset(BASE_DIR)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

start = time()
for epoch in range (1):
    print('EPOCH {}:'.format(epoch + 1))

    for step, (image, label) in enumerate(train_dl):
        image = image.cuda()
        label = label.cuda()

        optimizer.zero_grad()

        outputs = model(image)
        # print(image.shape,outputs.shape, label.shape)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, label)
        loss.backward()
        
        # Adjust learning weights
        optimizer.step()

        # if step % 1 == 0:
        #     step_list.append(step_list[-1] + 1)
        #     viz.line(X=np.array([step_list[-1]]), Y=np.array([loss.item()]), win=win, update='append')
            
        if step % 100 == 0:    
            print('     step:{}, loss:{:.3f}, time:{:.3f} min'
                  .format(step, loss.item(), (time() - start) / 60))

    print(' ')
    torch.save(model.state_dict(), BASE_DIR + 'net{}-{:.3f}.pth'.format(epoch, loss))
