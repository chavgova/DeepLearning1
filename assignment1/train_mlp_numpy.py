################################################################################
# MIT License
#
# Copyright (c) 2025 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2025
# Date Created: 2025-10-28
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

import torch
import matplotlib.pyplot as plt


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions between 0 and 1,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    
    
    pred_labels =np.argmax(predictions, axis=1)
    if targets.ndim > 1:
        targets = np.argmax(targets, axis=1) 
    accuracy = np.mean(pred_labels==targets)  
    
    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def evaluate_model(model, data_loader):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset, 
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    num_correct =0.0
    num_examples = 0
  
    
    for x, y in data_loader:  
        preds = model.forward(x)  
        
        if y.ndim > 1:
            y_eval = np.argmax(y, axis=1)
        else:
            y_eval=y
            
        num_correct += np.sum(np.argmax(preds,axis=1) == y_eval)
        num_examples +=y_eval.shape[0] 
        
    avg_accuracy = num_correct/float(num_examples)  # correct/total accuracy calculation
    
    
    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_accuracy


def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
      logging_dict: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    
    n_inputs = 3*32*32
    n_classes = 10
    model = MLP(n_inputs=n_inputs, n_hidden=hidden_dims, n_classes=n_classes)
    loss_module = CrossEntropyModule()

    val_accuracies = []  # all of these are for collecting the stats we need later for the plots
    train_accuracies =[]
    train_losses = []

    best_val =  -1.0   # I have the habbit of saving the best model so I ususally add these 
    best_model = None

    for ep in range(epochs):
        epoch_losses =[ ]
        epoch_correct = 0
        epoch_total = 0

        for x, y in cifar10_loader['train']:
            out= model.forward(x)
            loss= loss_module.forward(out, y)
            epoch_losses.append(loss) 

            if y.ndim > 1:
                y_idx = np.argmax(y, axis=1)
            else:
                y_idx = y
                
            epoch_correct += np.sum(np.argmax(out, axis=1) == y_idx)
            epoch_total += y_idx.shape[0]

            dloss = loss_module.backward(out, y)
            model.backward(dloss)

            for m in model.modules:
                if hasattr(m, 'params') and m.params['weight'] is not None:
                    m.params['weight'] -= lr * m.grads['weight']
                    m.params['bias']-= lr * m.grads['bias']

        train_losses.append(float(np.mean(epoch_losses)))
        train_accuracies.append(epoch_correct / float(epoch_total))

        val_acc = evaluate_model(model, cifar10_loader['validation'])
        val_accuracies.append(val_acc)

        if val_acc > best_val:  
            best_val=val_acc
            best_model = deepcopy(model) # I have the habbit of saving the best model just in case although not needed here 

    test_accuracy = evaluate_model(best_model, cifar10_loader['test'])
    
    logging_dict = {
        'train_loss': train_losses,'train_acc': train_accuracies,
        'val_acc': val_accuracies, 'test_acc': test_accuracy}
    
    
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_dict


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    model, val_accuracies, test_accuracy, logging = train(**kwargs)
 
    epochs= np.arange(1, len(logging["train_loss"]) +1)
    plt.figure()
    plt.plot(epochs,logging["train_loss"])   
    plt.ylabel("Training Loss") 
    plt.xlabel("Epoch")
    plt.title("MLP Training loss ")  
    plt.grid(True) 
    plt.tight_layout() 
    plt.show() 
         
    plt.figure() 
    plt.plot(epochs, logging["train_acc"], label="Train")
    plt.plot(epochs, logging["val_acc"], label="Validation")   
    plt.ylabel("Accuracy")  
    plt.xlabel("Epoch")
    plt.title(f"Accuracy [best test acc={logging['test_acc']:.2f}]")
    plt.legend()  
    plt.grid(True) 
    plt.tight_layout()
    plt.show()



