import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf
from dataset.dataset import DataSet
from visualization.visualization import Visualization
from model.xception import Xception
from train.train import Train
'''
    This project uses tensorflow 2.1.0
'''

print(tf.__version__)

ds  = DataSet()

def plot_model():
    xception =  Xception()
    xception.plot_model()

def show_ds_info():
    ds.show_dataset_description()

def do_training(epochs, lr):
    train  = Train()
    train.train(epochs=epochs, lr=lr)


if __name__ == '__main__':
    # show_ds_info()
    # do_training(epochs=2, lr=1e-3)
    # plot_model()