from dataset.dataset import DataSet
from settings import model_dir
from model.xception import Xception

import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
import numpy as np

class Test:

    def __init__(self):
        self.ds = DataSet(batch_size=-1)

    def test(self):
        test = tfds.as_numpy(self.ds.load_testset())
       
       
        json_file = open(str(model_dir.joinpath('model.json')), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        
        model = tf.keras.models.model_from_json(loaded_model_json)
        model.load_weights(str(model_dir.joinpath('model.h5')))
        print("Loaded model from disk")

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        results = model.evaluate(test['image'], test['label'])
        print('test loss, test acc:', results)

        for i, x in enumerate(test['image']):
            pred = model.predict(np.array([x]))
            j = test['label'][i]
            print("{}: {}: max: {}".format (i, pred[0][j], np.max(pred[0])))
