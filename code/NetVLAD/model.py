import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tensorflow as tf

from torch.autograd import Variable
from torchvision.models import resnet18

from tensorflow import keras
from keras.preprocessing import image

import PIL
import torch
import torchvision

import matplotlib.pyplot as plt
from sklearn import svm, datasets, metrics

import netvlad
import hard_triplet_loss

class NetVLADModel:
    def __init__(self):
        # Discard layers at the end of base network
        self.encoder = resnet18(pretrained=True)
        self.base_model = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu,
            self.encoder.maxpool,
            self.encoder.layer1,
            self.encoder.layer2,
            self.encoder.layer3,
            self.encoder.layer4,
        )
        dim = list(self.base_model.parameters())[-1].shape[0]  # last channels (512)

        # Define model for embedding
        self.net_vlad = netvlad.NetVLAD(num_clusters=6, dim=dim, alpha=1.0)
        self.model = netvlad.EmbedNet(self.base_model, self.net_vlad).cuda()

        # Define loss
        self.criterion = hard_triplet_loss.HardTripletLoss(margin=0.1).cuda()
        self.epochs = 50
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.ToTensor()

        ])

        