import tensorflow as tf
from dataset.dataset import DataSet
import tensorflow_datasets as tfds
import numpy as np

class InceptionV4:

    def __init__(self, num_classes=100, batch_size=32, shape=(32,32,3)):
        self.inputs = tf.keras.layers.Input(shape=shape, batch_size=batch_size)
        hiddens = self.inceptionv4(self.inputs)
        hiddens = tf.keras.layers.Flatten() (hiddens)
        self.outputs = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax,
                  kernel_initializer=tf.initializers.GlorotUniform())(hiddens)

    def get_model(self):
        return tf.keras.Model(self.inputs, self.outputs)

    def inceptionv4(self, inputs):
        return self.conv2d(inputs=inputs, filters=3, kernel_size=3, padding='VALID')

    def conv2d(self, inputs, filters, kernel_size, strides=1, padding='SAME'):
        outputs = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                        strides=strides, padding=padding,
                        kernel_initializer=tf.initializers.GlorotUniform(),
                        use_bias=False)(inputs)
        return tf.keras.layers.LeakyReLU()(outputs)

if __name__ == '__main__':
    incenption = InceptionV4()

    tf.keras.utils.plot_model(incenption.get_model(), 'InceptionV4.png', show_shapes=True)