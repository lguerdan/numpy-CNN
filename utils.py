#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 20:54:11 2019

@author: lukeguerdan
"""

import numpy as np
import pandas as pd
from numpy import genfromtxt
import matplotlib.pyplot as plt

def plot_image(image_array, raw=False):
    dim = int(np.sqrt(image_array.size))
    
    if raw: 
        image_array = image_array.reshape((dim,dim), order='F')
    else:
        image_array = image_array.reshape((dim,dim))
    
    plt.imshow(image_array)
    
def col_to_row_form(features):
    dim, dimall = int(np.sqrt(features.size)), features.size
    
    features = features.reshape((dim,dim), order='F')
    features = features.reshape(1, dimall)
    return features


def performance_metrics(cnn):
    pred_scores_train,pred_scores_test  = cnn.get_model_accuracy()

    dfTrain = pd.DataFrame(pred_scores_train)
    dfTest = pd.DataFrame(pred_scores_test)
    
    print("\nTrain confusion:")
    print(pd.crosstab(dfTrain.predicted, dfTrain.actual))
    print("\n\nTest confusion: ")
    print(pd.crosstab(dfTest.predicted, dfTest.actual))
    
    train_acc = (dfTrain.predicted == dfTrain.actual).mean()
    test_acc = (dfTest.predicted == dfTest.actual).mean()
    
    print("\n\nTrain accuary: " + str(train_acc))
    print("Test accuary: " + str(test_acc))
    