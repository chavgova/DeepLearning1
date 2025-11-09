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
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization. 
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """

        # Note: For the sake of this assignment, please store the parameters
        # and gradients in this format, otherwise some unit tests might fail.
        self.params = {'weight': None, 'bias': None} # Model parameters
        self.grads = {'weight': None, 'bias': None} # Gradients

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        std = np.sqrt(2.0/in_features)
        
        self.params['weight'] = np.random.randn(out_features, in_features)*std  # initializing the weights randomly
        self.params['bias'] = np.zeros((1, out_features), dtype=self.params['weight'].dtype) 

        self.grads['weight'] = np.zeros_like(self.params['weight'])  
        self.grads['bias'] = np.zeros_like(self.params['bias'])

        self._cache_x = None
        
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        self._cache_x=x
        out = x.dot(self.params['weight'].T) + self.params['bias'] # this is the linear forward calculation 
        
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        x = self._cache_x
        # for the back pass we need tje dx
        self.grads['weight'] = dout.T.dot(x)
        self.grads['bias'] = np.sum(dout, axis=0, keepdims=True)
        
        dx = dout.dot(self.params['weight'])
        
        
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        self._cache_x = None
        
        
        #######################
        # END OF YOUR CODE    #
        #######################


class ELUModule(object):
    """
    ELU activation module.
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        self._cache_x = x
        
        out_pos = x *(x > 0)  
        out_neg = self.alpha *(np.exp(x)-1.0) *(x<=0) 
        out = out_pos + out_neg  
        
        self._cache_out =out  
        
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        x = self._cache_x
        out = self._cache_out
        
        
        deriv = (x > 0).astype(x.dtype) + (x <= 0).astype(x.dtype)*(out+self.alpha) # we calc the derivative for the back passs
        dx = dout * deriv
        
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        self._cache_x = None
        self._cache_out = None
        
        #######################
        # END OF YOUR CODE    #
        #######################


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        x_shift = x-np.max(x, axis=1, keepdims=True)
        exps=np.exp(x_shift)
        out = exps / np.sum(exps, axis=1, keepdims=True)
        
        self._cache_out = out

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        y = self._cache_out
        s = np.sum(dout*y, axis=1, keepdims=True)
        dx = y * (dout-s)

        #######################
        # END OF YOUR CODE    #
        #######################

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        self._cache_out = None
        
        #######################
        # END OF YOUR CODE    #
        #######################


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        eps = 1e-12 # very small val for epsilon
        
        if y.ndim > 1: 
            y_idx = np.argmax(y, axis=1)
        else:
            y_idx = y
            
        N = x.shape[0]
        out = -np.mean(np.log(  x[np.arange(N), y_idx] + eps))

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        
        eps = 1e-12
        
        if y.ndim > 1:  
            y_idx = np.argmax(y, axis=1)   
        else:  
            y_idx = y
            
        N, C = x.shape  
        dx = np.zeros_like(x)  
        dx[np.arange(N),y_idx]= -1.0 / (N*( x[np.arange(N), y_idx] + eps)) # calc for the deriv
        
        #######################
        # END OF YOUR CODE    #
        #######################

        return dx
