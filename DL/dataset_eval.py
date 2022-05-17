import torch
from torch.utils.data import Dataset as dataset
from torchvision import transforms as tvtsf
import os
import numpy as np
import cv2

class Dataset(dataset):
    def __init__(self, BASE_DIR):
        self.image_list = []
        self.id = []
        self.classes_name = []
        
        impath = os.listdir(BASE_DIR + 'test')
        self.classes_name =  os.listdir(BASE_DIR + 'test')
        print(self.classes_name)

        for path in impath:

            class_path = BASE_DIR + "/test/" + path
            image_names= os.listdir(class_path)
            
            for image in image_names:
                self.image_list.append(class_path+'/'+ image)
                for i in range(0, len(self.classes_name)):
                    if path == self.classes_name[i]:
                        self.id.append(i)
                        # print(self.classes_name[i])

    def __getitem__(self, index):
        # print(self.train_image_list[index], self.train_id[index], index)

        image = cv2.imread(self.image_list[index])
        image = cv2.resize(image, (224,224), interpolation=cv2.INTER_CUBIC)

        image = np.moveaxis(image,2,0)
        label = self.id[index]

        image = torch.FloatTensor(image)
        # label = torch.FloatTensor(label)

        return image, label

    def __len__(self):
        return len(self.image_list)