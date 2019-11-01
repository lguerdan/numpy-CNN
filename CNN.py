#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 17:43:26 2019

@author: lukeguerdan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 19:09:47 2019

@author: lukeguerdan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_provider import DataProvider 
from utils import * 
from timeit import default_timer as timer
from scipy.signal import correlate2d 

class ConvNeuralNetwork:
    
    def __init__(self, part='2', img_width=28, filter_width=28, num_filters=2, num_classes=2, alpha=.01,
                 activation_function='sigmoid', relu_alpha=0, sig_lambdas=(1,1,1), subset_size=1, tanh_lambda=1):
        
        self.part = part
        if self.part == '2':
            self.filter_width=28
            self.num_filters=2
            num_classes=2
            train_dir = '../data/part2/train/*'
            test_dir = '../data/part2/train/*'
        
        if self.part == '3a' or part == '3b':
            self.filter_width=7
            self.num_filters=16
            num_classes=10
            train_dir = '../data/part3/train/*'
            test_dir = '../data/part3/train/*'
                
        
        self.img_width = img_width
 
        self.output_dim = num_classes
        self.alpha = alpha
        self.activation_function = activation_function
        
        self.relu_alpha = relu_alpha
        self.sig_lambdas = sig_lambdas
        self.tanh_lambda = tanh_lambda
        
        #computed properties
        self.conv_mat_H = np.power((img_width - self.filter_width + 1),2)  #number kernel positions
        self.conv_mat_S = img_width - self.filter_width                    #space between kerel and outside of image
        self.conv_output_dim = self.conv_mat_H * self.num_filters
        
        #create data provider to feed in data
        self.dp = DataProvider(train_dir, test_dir, num_classes, subset_size)
        
        if part == '2' or part == '3a':        
            self.init_weights_3A() 
        else: 
            self.init_weights_3B() 

           
    def train(self, epochs, shuffle, dp=False):
        
        epoch_sample_errors = []
        epoch_mses_train = []
        epoch_mses_test = []
        epoch_gradient_mags = []
        sample_gradient_mags = []
        
        if not dp: 
            dp = self.dp

        for features,target,epoch_over in dp.enumerate_dataset(train=True, epochs=epochs, shuffle=shuffle):
            
            if self.part != '3b':
                sample_error = self.forward_pass_3A(features, target)
                sample_gradient = self.backward_pass_3A(features, target)
            else: 
                sample_error = self.forward_pass_3B(features, target)
                sample_gradient = self.backward_pass_3B(features, target)   
            
            epoch_sample_errors.append(sample_error)
            sample_gradient_mags.append(sample_gradient)

            if epoch_over:
                if self.part != '2':
                    print('Epoch {} complete'.format(len(epoch_mses_train))) 
                
                test_mse = self.evaluate()
                
                epoch_mses_train.append(pd.Series(epoch_sample_errors).mean())
                epoch_gradient_mags.append(pd.Series(sample_gradient_mags).mean())
                epoch_mses_test.append(test_mse)
                
                epoch_sample_errors = []
                sample_gradient_mags = []
                
        return epoch_mses_train, epoch_mses_test, epoch_gradient_mags
    
    
    ############# Funcitons for Part 2 B ######################################
    def init_weights_3A(self):
        '''Initialize all weights and biases in the network, will add options for different strategies'''
        wConvVar = np.power(self.conv_mat_H , -.5)
        self.WConv = np.random.normal(0, wConvVar, size=(self.filter_width, self.filter_width, self.num_filters))
        self.bConv = np.random.normal(0, wConvVar, size=(self.num_filters, 1))
        
        w2Var = np.power(self.conv_output_dim, -.5)
        self.W2 = np.random.normal(0, w2Var, (self.output_dim, self.conv_output_dim))
        self.b2 = np.random.normal(0, w2Var, (self.output_dim, 1))
        
        #Create Toeplitz Matrix for backprop and deconv
        self.WConvC = self.filter_to_C(self.WConv)
                
    
    def forward_pass_3A(self, features, target):
        '''Compute forward pass over this data sample, returning SSE for pass'''
        features_sq = features.reshape(self.img_width, self.img_width)
        self.ConvY = np.zeros((self.num_filters, self.conv_mat_H))

        for filter in range(self.num_filters):
            activation = correlate2d(features_sq, self.WConv[:,:,filter], mode='valid')
            self.ConvY[filter, :] = activation.reshape(1, self.conv_mat_H)
        
        self.ConvY = self.activation(self.ConvY + self.bConv, self.sig_lambdas[0])
        self.ConvY = self.ConvY.reshape(self.ConvY.shape[0] * self.ConvY.shape[1],1)
        
        self.Y2 = self.W2.dot(self.ConvY)
        self.Y2 = self.activation(self.Y2 + self.b2, self.sig_lambdas[1])

        self.error = target - self.Y2
        return np.sum(np.square(self.error)) / 2
   
        
    def backward_pass_3A(self, features, target):
        '''Compute backward pass over this data sample'''
        
        #Ouput layer deltas
        d2 = np.multiply(self.error, self.activation_p(self.Y2, self.sig_lambdas[1]))
        
        #Connected layer weight gradients
        W2_grad = np.outer(d2, self.ConvY)
        W2_update = np.multiply(self.alpha, W2_grad)
        b2_update = np.multiply(self.alpha, d2)
        
        #Conv output deltas
        dConv = np.multiply(self.activation_p(self.ConvY, self.sig_lambdas[0]), np.transpose(self.W2).dot(d2))
        dConv = dConv.reshape(self.conv_mat_H, self.num_filters)
        self.dConv = dConv
        
        #Conv weight gradients
        input_tiled = self.create_input_tiled(features.T).T
        WConv_grads = np.matmul(input_tiled, dConv).reshape(self.filter_width, self.filter_width, self.num_filters)
        WConv_update = np.multiply(self.alpha, WConv_grads)
        self.WConv_grads = WConv_grads
        
        #Conv bias
        bConv_grads = np.sum(dConv.T,axis=0)
        bConv_update = np.multiply(self.alpha, bConv_grads)
          
        #Perform weight updates
        self.W2 = self.W2 + W2_update
        self.b2 = self.b2 + b2_update
        
        self.WConv = self.WConv + WConv_update
        self.bConv = self.bConv + bConv_update
        
        return np.sum(np.square(dConv))

    
    ############End Functions for part 3 A ####################################    
    ############# Funcitons for Part 3 B ######################################
    
    def init_weights_3B(self):
        '''Initialize all weights and biases in the network, will add options for different strategies'''
        
        wConvVar = np.power(self.conv_mat_H , -.5)
        self.WConv = np.random.normal(0, wConvVar, size=(self.filter_width, self.filter_width, self.num_filters))
        self.bConv = np.random.normal(0, wConvVar, size=(self.num_filters, 1))
        
        w2Avar = np.power(self.conv_output_dim, -.5)
        self.W2a = np.random.normal(0, w2Avar, (128, self.conv_output_dim))
        self.b2a = np.random.normal(0, w2Avar, (128, 1))
        w2dim = 128
    
        
        w2Var = np.power(w2dim, -.5)  
        self.W2 = np.random.normal(0, w2Var, (self.output_dim, w2dim))
        self.b2 = np.random.normal(0, w2Var, (self.output_dim, 1))
        
        #Create Toeplitz Matrix for backprop and deconv
        self.WConvC = self.filter_to_C(self.WConv)      
    
    def forward_pass_3B(self, features, target):
        '''Compute forward pass over this data sample, returning SSE for pass'''

        features_sq = features.reshape(self.img_width, self.img_width)
        self.ConvY = np.zeros((self.num_filters, self.conv_mat_H))

        for filter in range(self.num_filters):
            activation = correlate2d(features_sq, self.WConv[:,:,filter], mode='valid')
            self.ConvY[filter, :] = activation.reshape(1, self.conv_mat_H)
        
        self.ConvY = self.activation(self.ConvY + self.bConv, self.sig_lambdas[0])
        self.ConvY = self.ConvY.reshape(self.ConvY.shape[0] * self.ConvY.shape[1],1)
         
        self.Y2a = self.W2a.dot(self.ConvY)
        self.Y2a = self.activation(self.Y2a + self.b2a, self.sig_lambdas[1])
      
        self.Y2 = self.W2.dot(self.Y2a)
        self.Y2 = self.activation(self.Y2 + self.b2, self.sig_lambdas[2])

        self.error = target - self.Y2
        return np.sum(np.square(self.error)) / 2
        
        
    def backward_pass_3B(self, features, target):
        '''Compute backward pass over this data sample'''
        
        #Ouput layer deltas
        d2 = np.multiply(self.error, self.activation_p(self.Y2, self.sig_lambdas[1]))
        
        W2_grad = np.outer(d2, self.Y2a) 
        W2_update = np.multiply(self.alpha, W2_grad)
        b2_update = np.multiply(self.alpha, d2)  
        d2a = np.multiply(self.activation_p(self.Y2a, self.sig_lambdas[0]), np.transpose(self.W2).dot(d2))
         
        #Connected layer weight gradients
        W2a_grad = np.outer(d2a, self.ConvY)
        W2a_update = np.multiply(self.alpha, W2a_grad)
        b2a_update = np.multiply(self.alpha, d2a)
        
        #Conv output deltas
        dConv = np.multiply(self.activation_p(self.ConvY, self.sig_lambdas[0]), np.transpose(self.W2a).dot(d2a))
        dConv = dConv.reshape(self.conv_mat_H, self.num_filters)
        self.dConv = dConv
        
        #Conv weight gradients
        input_tiled = self.create_input_tiled(features.T).T
        WConv_grads = np.matmul(input_tiled, dConv).reshape(self.filter_width, self.filter_width, self.num_filters)
        WConv_update = np.multiply(self.alpha, WConv_grads)
        self.WConv_grads = WConv_grads
        
        #Conv bias
        bConv_grads = np.sum(dConv.T,axis=0)
        bConv_update = np.multiply(self.alpha, bConv_grads)
          
        #Perform weight updates
        self.W2 = self.W2 + W2_update
        self.b2 = self.b2 + b2_update
        
        self.W2a = self.W2a + W2a_update
        self.b2a = self.b2a + b2a_update
        
        self.WConv = self.WConv + WConv_update
        self.bConv = self.bConv + bConv_update
        
        return np.sum(np.square(dConv))
    
    ####################### End functions for part 3 B ########################################################

    def evaluate(self):
        
        mses = [] 
        for features,target,epoch_over in self.dp.enumerate_dataset(train=False, epochs=1, shuffle=False): 
            
            if self.part != '3b': 
                sample_error = self.forward_pass_3A(features, target)
            else: 
                sample_error = self.forward_pass_3B(features, target)
            
            mses.append(sample_error) 
        
        return pd.Series(mses).mean()
       
    
    def get_model_accuracy(self):
         
        predicted_train = []
        actual_train = []
        predicted_test = []
        actual_test = []
        
        for features,target,epoch_over in self.dp.enumerate_dataset(train=True, epochs=1, shuffle=False):
            
            if self.part != '3b': 
                self.forward_pass_3A(features, target)
            else: 
                self.forward_pass_3B(features, target)
             
            predicted_train.append(self.Y2.argmax())
            actual_train.append(target.argmax())

        for features,target,epoch_over in self.dp.enumerate_dataset(train=False, epochs=1, shuffle=False):
            
            if self.part != '3b': 
                self.forward_pass_3A(features, target)
            else: 
                self.forward_pass_3B(features, target)
                
            predicted_test.append(self.Y2.argmax())
            actual_test.append(target.argmax())
        

        prediction_scores_train = {'predicted': predicted_train,
                             'actual' : actual_train}  
        prediction_scores_test = {'predicted': predicted_test, 
                             'actual': actual_test}
        
        return prediction_scores_train, prediction_scores_test

    
    def filter_to_C(self, WConv):
        '''Helper method: Converts a square filter to a staggered matrix C'''
        conv_vec_buff_h = np.zeros((self.filter_width, self.conv_mat_S, self.num_filters))
        conv_vec = np.concatenate([WConv,conv_vec_buff_h], axis=1)
        conv_vec_buff_v = np.zeros((self.conv_mat_S, conv_vec.shape[1], self.num_filters))
        conv_vec = np.concatenate([conv_vec, conv_vec_buff_v])
        conv_vec = conv_vec.reshape(1,np.power(self.img_width, 2), self.num_filters)
        conv_mat = np.tile(conv_vec, (self.conv_mat_H,1,1 ))
        
        #Create standard conv matrix
        for i in range(1,self.conv_mat_H):
            if i % (self.conv_mat_S + 1 ) == 0:
                conv_mat[i:,:,:] = np.roll(conv_mat[i:,:,:], self.filter_width, axis=1)
            else:
                conv_mat[i:,:,:] = np.roll(conv_mat[i:,:,:], 1, axis=1)
        
        return conv_mat
    
    def create_input_tiled(self, features):
        '''Creates a tiled input matrix to use for back proping weignts'''
        
        input_tiled = np.tile(features, (1,self.WConvC.shape[0])).T
        input_tiled = input_tiled[self.WConvC[:,:,0] != 0].reshape(self.conv_mat_H, self.filter_width ** 2)
        return input_tiled
    
    def perform_deconvolution(self):
        
        for imclass in range(self.dp.deconv_samples.shape[0]):
            
            features, target = self.dp.deconv_samples[imclass, 1:], self.dp.deconv_samples[imclass, 0]
            print('Image class: '.format(target))
            if self.part != '3b': 
                self.forward_pass_3A(features, target)
            else: 
                self.forward_pass_3B(features, target)
            
        
            WConvC = self.filter_to_C(self.WConv)
    
            for filter in range(self.num_filters):
                dConv = self.dConv[:,filter]
                WConvCs = WConvC[:,:,filter]
                deconv_activations = dConv.dot(WConvCs)
                deconv_activations = deconv_activations.reshape(self.img_width, self.img_width)
                plt.figure()
                plt.imshow(deconv_activations)
                plt.title('Activations for filter {}'.format(filter))
    
    
    def activation(self, array, sig_lambda=1, tanh_lambda=1):
        if self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(sig_lambda * - array))

        elif self.activation_function == 'tanh':
            num = (np.exp(self.tanh_lambda * array) - np.exp(self.tanh_lambda * -array))
            denom = (np.exp(self.tanh_lambda * array) + np.exp(self.tanh_lambda * -array))
            return num / denom
            
        elif self.activation_function == 'relu':
            return abs(array) * (array > 0)
        
    
    def activation_p(self, array, sig_lambda=1):
        
        if self.activation_function == 'sigmoid':
            return sig_lambda * array * (1.0 - array)

        elif self.activation_function == 'tanh':
            return self.tanh_lambda * 1 - np.power(array, 2)
        
        elif self.activation_function == 'relu':
            return np.where(array <= 0, self.relu_alpha, 1)    
    