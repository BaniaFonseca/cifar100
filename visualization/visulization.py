from dataset.dataset import DataSet

import tensorflow_datasets as tfds
import numpy as np
from matplotlib import pyplot as plt

class Visualization(DataSet):

    def __init__(self):
        super(Visualization, self).__init__()
        self.train, self.test = self.get_dataset()

    def view_images(self, dataset, cols=6, rows=2, name=""):
        plt.figure(figsize=(32, 32))

        for examples in tfds.as_numpy(dataset):
            for n in range(rows*cols):
                plt.subplot(rows, cols, n+1)
                plt.imshow(examples['image'][n])
                plt.colorbar()
                plt.xticks([])
                plt.yticks([])
                plt.xlabel(examples['label'][n])
            break
        plt.show()

    def view_test_images(self, cols=6, rows=2):
        test = self.test.batch(rows*cols).repeat(1)
        self.view_images(dataset=test, rows=rows, cols=cols, name="train images")

    def view_train_images(self, cols=6, rows=2):
        train = self.train.batch(rows*cols).repeat(1)
        self.view_images(dataset=train, rows=rows, cols=cols, name="test images")

if __name__ == '__main__':
    vs = Visualization()