import torch
from torch.utils.data import Dataset as dataset
import os
import SimpleITK as sitk
import numpy as np
import cv2

class Dataset(dataset):
    def __init__(self, BASE_DIR):
        self.train_image_list = []
        self.train_id = []
        
        impath = os.listdir(BASE_DIR + 'test')

        for path in impath:
            image_path=[]
            image_id=[]

            class_path = BASE_DIR + "/test/" + path
            image_names= os.listdir(class_path)
            
            for image in image_names:
                image_path.append(class_path+'/'+ image)
                image_id.append(int(path))
                
            # Choose randomly 1000 images for testing and 9000 images for training
            y=np.arange(1000)
            np.random.shuffle(y)
            # print(y.shape, len(image_path), len(image_id))
            # self.train_image_list += [self.image_path[i] for i in y[0:6000]]
            # self.train_id += [self.image_id[i] for i in y[0:6000]]
            
            for i in range (len(y)):
                self.train_image_list.append(image_path[i])
                self.train_id.append(image_id[i])

    def __getitem__(self, index):
        # print(self.train_image_list[index], self.train_id[index], index)

        image = cv2.imread(self.train_image_list[index])
        image = cv2.resize(image, (224,224), interpolation=cv2.INTER_CUBIC)
        image = np.moveaxis(image,2,0)
        label = self.train_id[index]

        image = torch.FloatTensor(image)
        # label = torch.FloatTensor(label)

        return image, label

    def __len__(self):
        return len(self.train_image_list)