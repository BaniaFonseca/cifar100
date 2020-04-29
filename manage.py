'''
    This project uses tensorflow 2.1.0
'''
import tensorflow as tf
from dataset.dataset import DataSet
from visualization.visualization import Visualization
print(tf.__version__)

x = tf.Variable(7, name='x')
y = tf.Variable(7, name='y')


@tf.function
def sum(z):
    return z*(x+y)

@tf.function
def forward():
    return sum(2)


out = forward()
print(out.numpy())

ds  = DataSet()
ds.show_dataset_description()

vs = Visualization()
vs.view_train_images()
vs.view_test_images()
