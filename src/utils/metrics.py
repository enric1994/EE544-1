
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model

from sklearn import metrics
import numpy as np

def get_metrics(experiment,image_size):
    test_path = '/data/resized_{}/test'.format(image_size)
    batch_size = 64

    test_datagen = ImageDataGenerator(
            rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
            test_path,
            target_size=(299, 299),
            batch_size=batch_size,
            class_mode='binary')


    model = load_model('/code/checkpoints/{}.weights'.format(experiment))

    predictions = model.predict_generator(test_generator)

    val_preds = (predictions < 0.5).astype(np.int)
    val_trues = test_generator.classes
    cm = metrics.confusion_matrix(val_trues, val_preds)

    labels = test_generator.class_indices.keys()
    precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(val_trues, val_preds, labels=[0,1])
    print('Precision for "No Finding": {}'.format(precisions[0]))
    print('Precision for "Cardiomegaly": {}'.format(precisions[1]))
    print('Recall for "No Finding": {}'.format(recall[0]))
    print('Recall for "Cardiomegaly": {}'.format(recall[1]))
    print('F1 score for "No Finding": {}'.format(f1_score[0]))
    print('F1 score for "Cardiomegaly": {}'.format(f1_score[1]))
    print('Confusion matrix:')
    print(cm)

get_metrics('3.14.2',299)