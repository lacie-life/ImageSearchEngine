import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import cv2

BASE_DIR = 'D:/CV Course/FashionMNIST/data/'
from collections import OrderedDict
new_state_dict = OrderedDict()
state_dict = torch.load(BASE_DIR + 'net14-0.012.pth')
# print(state_dict.keys())
for k, v in state_dict.items():
    name = 'module.' + k # remove `module.`
    new_state_dict[name] = v
# print(new_state_dict.keys())

model = torch.nn.DataParallel(torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=False)).cuda()
model.load_state_dict(new_state_dict)
model.eval()

test_img = cv2.imread(BASE_DIR + 'train/0/1.jpg')
# test_img = cv2.resize(test_img, (224,224), interpolation=cv2.INTER_CUBIC)
test_img = np.moveaxis(test_img,2,0)
test_img = torch.FloatTensor(test_img).cuda().unsqueeze(0)
print(test_img.shape)

outputs = model(test_img)
print(outputs.shape)
print(outputs)