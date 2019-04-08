
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model

from sklearn import metrics
import numpy as np

import keras.backend as K
K.set_floatx('float16')


def get_metrics(experiment,image_size):
    test_path = '/data/resized_{}/test'.format(image_size)
    batch_size = 64

    test_datagen = ImageDataGenerator(
            rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
            test_path,
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode='binary',
            shuffle = False)


    model = load_model('/code/checkpoints/{}.weights'.format(experiment))

    predictions = model.predict_generator(test_generator)

    val_preds = (predictions >= 0.5).astype(np.int)
    val_trues = test_generator.classes
    cm = metrics.confusion_matrix(val_trues, val_preds)

    labels = test_generator.class_indices.keys()
    print('Accuracy: {}'.format(metrics.accuracy_score(val_trues, val_preds)))
    print('Precision: {}'.format(metrics.precision_score(val_trues, val_preds)))
    print('Recall: {}'.format(metrics.recall_score(val_trues, val_preds)))
    print('F1 score: {}'.format(metrics.f1_score(val_trues, val_preds)))
    print('Confusion matrix:')
    print(cm)

get_metrics('1.4.23',224)