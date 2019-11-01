#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 20:30:01 2019

@author: lukeguerdan
"""
import numpy as np
import pandas as pd
from numpy import genfromtxt
from utils import plot_image, col_to_row_form
import glob

class DataProvider:
    def __init__(self, train_dir, test_dir, num_classes, subset_size):
        train_files = glob.glob(train_dir)
        test_files = glob.glob(test_dir)
        
        self.dataset_train = self.generate_dataset(train_files, subset_size)[0]
        self.dataset_test, self.deconv_samples = self.generate_dataset(test_files, subset_size)
        self.num_classes = num_classes
    
    @staticmethod
    def generate_dataset(fnames, subset_size):
        '''Reads list of filenames and imports as tabular dataset'''
        
        dataset_classes = []
        dataset_examples = []
        
        for file in fnames:  
            label = int(file.split('_')[1])
            dataset = genfromtxt(file, delimiter=',' , dtype=np.float64)
            num_records = int(dataset.shape[0] * subset_size)
            dataset = dataset[0:num_records, :]
            
            #Normalize images
            dataset = np.subtract(dataset.T, np.mean(dataset, axis=1)).T
            dataset = np.divide(dataset.T, np.std(dataset, axis=1)).T
            dataset = np.concatenate([np.ones((num_records,1)) * label, dataset], axis=1)
            
            dataset_examples.append(np.expand_dims(dataset[0,:], axis=0))
            dataset_classes.append(dataset)
            
        dataset_all_classes = np.concatenate(dataset_classes)
        dataset_deconv_sample = np.concatenate(dataset_examples)
        np.random.shuffle(dataset_all_classes)
        
        return dataset_all_classes, dataset_deconv_sample
    
    def enumerate_dataset(self, train, epochs, shuffle):
        '''Generator to enumerate over the dataset
            train: True/False
            epochs: number of times to enumerate over data
            shuffle: shuffle between epochs
            yield: features, label, status (True if epoch complete, false otherwise)
        '''
        
        samples_per_epoch = self.dataset_train.shape[0] if train else self.dataset_test.shape[0]
        sample_permutation = [i for i in range(0,samples_per_epoch)]
        np.random.shuffle(sample_permutation)
        
        for i in range(epochs):      
            for j in range(samples_per_epoch):
                
                if train: 
                    features, label = self.dataset_train[sample_permutation[j],1:], self.dataset_train[sample_permutation[j],0]
                else:
                    features, label = self.dataset_test[sample_permutation[j],1:], self.dataset_test[sample_permutation[j],0]
                                
                features = col_to_row_form(features)                
                label_one_hot = self.to_one_hot(label, self.num_classes)
                
                epochEnds = False if j < samples_per_epoch - 1 else True
                yield features, label_one_hot, epochEnds
            
            if shuffle:
                np.random.shuffle(sample_permutation)
            
    def get_example(self):
        sample = np.random.randint(0,self.dataset_train.shape[0])    
        features, label = self.dataset_train[sample,1:], self.dataset_train[sample,0]
    
        features = np.expand_dims(features, axis=1).T
        label_one_hot = self.to_one_hot(label, 2)
        return features, label_one_hot
  
    
    @staticmethod
    def to_one_hot(target, nclasses):

        #If part A articicially convert 
        if nclasses == 2:
            if int(target) == 1:
                target = 0
            elif int(target) == 3:
                target = 1 
        
        return np.expand_dims(np.eye(nclasses)[int(target)].T, 1)
