import tensorflow_datasets as tfds
import numpy as np

class DataSet:

    def __init__(self):
        self.cifar100 = tfds.builder('cifar100')
        self.train_dataset = None
        self.test_dataset = None

    def download_and_prepare(self):
        self.cifar100 = self.cifar100.download_and_prepare()

    def get_dataset(self):
        self.cifar100 = self.cifar100.as_dataset(shuffle_files=True, )
        self.train_dataset, self.test_dataset = self.cifar100['train'], self.cifar100['test']
        return (self.train_dataset, self.test_dataset)

if __name__  == '__main__':
    dataset  = DataSet()

    print("========== dataset Descriptiom ================")

    print("image shape : {}".format(dataset.cifar100.info.features['image'].shape))
    print("numbers of labels : {}".format(dataset.cifar100.info.features['label'].num_classes))
    print("numbers of examples train : {}".format(dataset.cifar100.info.splits['train'].num_examples))
    print("numbers of examples test : {}".format(dataset.cifar100.info.splits['test'].num_examples))
