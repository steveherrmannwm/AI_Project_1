'''
Created on Aug 28, 2016

@author: km_dh
'''
import os, struct
import numpy as np
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros

class MNISTcontrol(object):
    '''
    Helper methods for using MNIST dataset
    '''

    def __init__(self, path="."):
        '''
        Constructor
        '''
        self.path = path
        
    def load_mnist(self,dataset="training", digits=np.arange(10)):
        """
        Loads MNIST files into 3D numpy arrays
        Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
        """
        path = self.path
        if dataset == "training":
            fname_img = os.path.join(path, 'digits-train/train-images-idx3-ubyte')
            fname_lbl = os.path.join(path, 'digits-train/train-labels-idx1-ubyte')
        elif dataset == "test":
            fname_img = os.path.join(path, 'digits-test/t10k-images-idx3-ubyte')
            fname_lbl = os.path.join(path, 'digits-test/t10k-labels-idx1-ubyte')
        else:
            raise ValueError("dataset must be 'test' or 'training'")

        flbl = open(fname_lbl, 'rb')
        magic_nr, size = struct.unpack(">II", flbl.read(8))
        lbl = pyarray("b", flbl.read())
        flbl.close()

        fimg = open(fname_img, 'rb')
        magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = pyarray("B", fimg.read())
        fimg.close()

        ind = [ k for k in range(size) if lbl[k] in digits ]
        N = len(ind)

        images = zeros((N, rows, cols), dtype=uint8)
        labels = zeros((N, 1), dtype=int8)
        for i in range(len(ind)):
            images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
            labels[i] = lbl[ind[i]]

        return images, labels