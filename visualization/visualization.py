import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
from dataset.dataset import DataSet

class Visualization(DataSet):

    def __init__(self, rows=2, cols=8):
        super(Visualization, self).__init__(batch_size=(rows*cols))
        self.train, self.test = self.load_trainset(), self.load_testset()
        self.rows = rows
        self.cols = cols

    def view_test_images(self):
        self.view_images(dataset=self.test)

    def view_train_images(self):
        self.view_images(dataset=self.train)

    def view_images(self, dataset):
        plt.figure(figsize=(32, 32))
        for batch in tfds.as_numpy(dataset):
            for n, example in enumerate(zip(batch['image'], batch['label']), 1):
                image, label = example
                plt.subplot(self.rows, self.cols, n)
                plt.imshow(image)
                plt.xticks([])
                plt.yticks([])
                plt.xlabel(label)
            break
        plt.show()

if __name__ == '__main__':
    vs = Visualization()
    vs.view_train_images()
    vs.view_test_images()
