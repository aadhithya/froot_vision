import numpy as np


class FrootVision:
    
    def __init__(self):
        print('Hello')
    
    def load_data(self):
        self.X = np.load('data/X_train.npy')
        self.Y = np.load('data/Y_train.npy')
        self.X_test = np.load('data/X_val.npy')
        self.Y_test = np.load('data/Y_val.npy')
        print(self.X.shape)
        print(self.Y.shape)
        print(self.X_test.shape)
        print(self.Y_test.shape)

fv = FrootVision()
fv.load_data()