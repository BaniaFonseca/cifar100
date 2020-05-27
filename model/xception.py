import tensorflow as tf
from pathlib import Path
from settings import model_dir

class Xception:

    def __init__(self, num_classes=100):
        self.inputs = tf.keras.layers.Input(shape=(32, 32, 3))
        hiddens = self.build_xception(self.inputs)
        hiddens = tf.keras.layers.Flatten() (hiddens)
        self.outputs = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax,
                  kernel_initializer=tf.initializers.GlorotUniform())(hiddens)

    def get_model(self):
        return tf.keras.Model(self.inputs, self.outputs)

    def build_xception(self, inputs):
        inputs = self.build_entry_flow(inputs)  
        inputs = self.build_middle_flow(inputs)
        inputs = self.build_exit_flow(inputs)  
        return inputs

    def build_entry_flow(self, inputs):
        inputs = self.do_conv2d(inputs=inputs, filters=32, strides=2 , kernel_size=3)
        inputs = self.do_conv2d(inputs=inputs, filters=64, kernel_size=3)
        
        short_cut = self.do_conv2d(inputs=inputs, filters=128, kernel_size=3, 
                                    useReLu=False, useMaxPooling=False)
        inputs = self.do_separableconv2d(inputs=inputs, filters=128, kernel_size=3)
        inputs = self.do_separableconv2d(inputs=inputs, filters=128, kernel_size=3,
                                        useReLu=False, useMaxPooling=False)
        inputs = self.add([inputs, short_cut])
        
        inputs = self.do_separableconv2d(inputs=inputs, filters=256, kernel_size=3)
        inputs = self.do_separableconv2d(inputs=inputs, filters=256, kernel_size=3, 
                                        useReLu=False, useMaxPooling=False)
        short_cut = self.do_conv2d(inputs=short_cut, filters=256, kernel_size=1, 
                                    useReLu=False, useMaxPooling=False) 
        inputs = self.add([inputs, short_cut])

        inputs = self.do_separableconv2d(inputs=inputs, filters=768, kernel_size=3)
        inputs = self.do_separableconv2d(inputs=inputs, filters=768, kernel_size=3,
                                        useReLu=False)
        short_cut = self.do_conv2d(inputs=short_cut, filters=768, kernel_size=1, 
                                    useReLu=False) 
        inputs = self.add([inputs, short_cut])

        return inputs

    def build_middle_flow(self, inputs):
        short_cut = inputs
        
        for __ in range(8):
            inputs = self.do_separableconv2d(inputs=inputs, filters=768, kernel_size=3)
            inputs = self.do_separableconv2d(inputs=inputs, filters=768, kernel_size=3)
            inputs = self.do_separableconv2d(inputs=inputs, filters=768, kernel_size=3, useReLu=False)
            short_cut = self.do_conv2d(inputs=short_cut, filters=768, kernel_size=1, useReLu=False) 
            inputs = self.add([inputs, short_cut])
            
        return inputs

    def build_exit_flow(self, inputs):
        short_cut = inputs

        inputs = self.do_separableconv2d(inputs=inputs, filters=512, kernel_size=3)
        inputs = self.do_separableconv2d(inputs=inputs, filters=512, kernel_size=3, useReLu=False)
        short_cut = self.do_conv2d(inputs=short_cut, filters=512, kernel_size=1, useReLu=False) 
        inputs = self.add([inputs, short_cut], useReLu=False)
        
        inputs = self.do_separableconv2d(inputs=inputs, filters=512, kernel_size=3)
        inputs = self.do_separableconv2d(inputs=inputs, filters=512, kernel_size=3)
        
        return inputs

    def do_conv2d(self, inputs, filters, kernel_size, strides=1, pool_size=3, 
                pool_strides=2, padding='SAME', useReLu=True, useMaxPooling=False):
        outputs = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                        strides=strides, padding=padding,
                        kernel_initializer=tf.initializers.GlorotUniform(),
                        use_bias=False)(inputs)
        outputs = tf.keras.layers.BatchNormalization(center=False) (outputs)

        if useMaxPooling:
            outputs = tf.keras.layers.MaxPool2D(pool_size=pool_size, 
                            strides = pool_strides) (outputs)

        if useReLu:
            return tf.keras.layers.ReLU()(outputs)
        else:
            return outputs

    def do_separableconv2d(self, inputs, filters, kernel_size, strides=1,
                            pool_size=3, pool_strides=2, padding='SAME', 
                            useReLu=True, useMaxPooling=False):
        outputs = tf.keras.layers.SeparableConv2D(filters=filters, kernel_size=kernel_size,
                        strides=strides, padding=padding,
                        use_bias=False)(inputs)
        outputs = tf.keras.layers.BatchNormalization(center=False) (outputs)    
        
        if useMaxPooling:
            outputs = tf.keras.layers.MaxPool2D(pool_size=pool_size, 
                            strides = pool_strides) (outputs)
        if useReLu:
            return tf.keras.layers.ReLU()(outputs)
        else:
            return outputs
    
    def add(self, inputs, useReLu=True):
        outputs = tf.keras.layers.Add() (inputs)
        
        if useReLu:
            return tf.keras.layers.ReLU()(outputs)
        else:
            return outputs
    
    def plot_model(self):
        model = self.get_model()
        model.summary()
        img_file = Path.joinpath(model_dir, 'Xception.png')
        tf.keras.utils.plot_model(model, to_file=img_file, show_shapes=True)