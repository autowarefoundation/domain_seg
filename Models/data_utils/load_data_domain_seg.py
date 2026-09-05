#! /usr/bin/env python3
import pathlib
import numpy as np
from PIL import Image
from .check_data import CheckData

class LoadDataDomainSeg():
    def __init__(self, labels_filepath, images_filepath, dataset_name=""):

        # Sort data and get list of input images and ground truth labels
        self.labels = sorted([f for f in pathlib.Path(labels_filepath).glob("*.png")])
        self.images = sorted([f for f in pathlib.Path(images_filepath).glob("*.jpg")])

        # Number of input images and ground truth labels
        self.num_images = len(self.images)
        self.num_labels = len(self.labels)

        # Performing sanity checks to ensure samples are correct in number
        checkData = CheckData(dataset_name, self.num_images, self.num_labels)

        # Lists to store train/val data
        self.train_images = []
        self.train_labels = []
        self.val_images = []
        self.val_labels = []
        
        # Number of train/val samples
        self.num_train_samples = 0
        self.num_val_samples = 0

        # If all checks have passed, get samples and assign to train/val splits
        if (checkData.getCheck()):
            # Assigning ground truth data to train/val split
            for count in range (0, self.num_images):
        
                if((count+1) % 50 == 0):
                    self.val_images.append(str(self.images[count]))
                    self.val_labels.append(str(self.labels[count]))
                    self.num_val_samples += 1 
                else:
                    self.train_images.append(str(self.images[count]))
                    self.train_labels.append(str(self.labels[count]))
                    self.num_train_samples += 1

    # Getting number of train/val samples
    def getItemCount(self):
        return self.num_train_samples, self.num_val_samples
    
    # Get training data in numpy format
    def getItemTrain(self, index):
        train_image = Image.open(str(self.train_images[index])).convert('RGB')
        train_mask = Image.open(str(self.train_labels[index]))
        train_ground_truth = np.expand_dims(train_mask, axis=-1)
        return  np.array(train_image), np.array(train_ground_truth)
    
    # Get training data in numpy format
    def getItemVal(self, index):
        val_image = Image.open(str(self.val_images[index])).convert('RGB')
        val_mask = Image.open(str(self.val_labels[index]))
        val_ground_truth = np.expand_dims(val_mask, axis=-1)
        return  np.array(val_image), np.array(val_ground_truth)