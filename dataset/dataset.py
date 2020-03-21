import tensorflow_datasets as tfds

class DataSet:

    def __init__(self, batch_size=32):
        self.cifar100 = tfds.builder('cifar100')
        self.dataset = self.cifar100.as_dataset(shuffle_files=True, batch_size=batch_size)

    def download_and_prepare(self):
        # download path: /home/fonseca/tensorflow_dataset/cifar100/3.0.0
        self.cifar100.download_and_prepare()

    def get_dataset(self):
        return (self.dataset['train'], self.dataset['test'])

    def show_dataset_description(self):
        print("========== dataset Description ================")

        print("image shape : {}".format(self.cifar100.info.features['image'].shape))
        print("numbers of labels : {}".format(self.cifar100.info.features['label'].num_classes))
        print("numbers of examples train : {}".format(self.cifar100.info.splits['train'].num_examples))
        print("numbers of examples test : {}".format(self.cifar100.info.splits['test'].num_examples))

if __name__  == '__main__':
    ds  = DataSet()
    ds.show_dataset_description()