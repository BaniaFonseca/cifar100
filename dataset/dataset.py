import tensorflow_datasets as tfds
from settings import *

class DataSet:

    def __init__(self, batch_size=512):
        download = lambda data_dir: not data_dir.joinpath('cifar100').exists()
        self.ds, self.ds_info = tfds.load(name='cifar100', 
                                            data_dir=data_dir, 
                                            batch_size=batch_size, shuffle_files=True,
                                            with_info=True, download=download)

    def load_trainset(self):
        return self.ds[tfds.Split.TRAIN]

    def load_testset(self):
            return self.ds[tfds.Split.TEST]

    def show_dataset_description(self):
        print("========== dataset Description ================")
        print("image shape : {}".format(self.ds_info.features['image'].shape))
        print("numbers of labels : {}".format(self.ds_info.features['label'].num_classes))
        print("numbers of examples train : {}".format(self.ds_info.splits['train'].num_examples))
        print("numbers of examples test : {}".format(self.ds_info.splits['test'].num_examples))

if __name__  == '__main__':
    ds  = DataSet()
    ds.show_dataset_description()