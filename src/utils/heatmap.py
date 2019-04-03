#Reference:
#https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.4-visualizing-what-convnets-learn.ipynb

import keras
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# import keras.backend as K
# K.set_floatx('float16')

def heatmap(model, img_path, layer='conv2d_2',iterations=30):

    # model = load_model('/code/checkpoints/{}.weights'.format(model_name))
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    image_output = model.output[:, np.argmax(preds[0])]
    last_conv_layer = model.get_layer(layer)
    grads = K.gradients(image_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for i in range(30):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img

    return superimposed_img

def bulk_heatmap(model_name):

    model = load_model('/code/checkpoints/{}.weights'.format(model_name))

    for root, dirs, files in os.walk("/data/resized_224/test/Cardiomegaly"):
        for name in files:
            img_path = os.path.join(root,name)
            for layer in range(9,13):
                # try:
                img = heatmap(model, img_path, 'conv2d_{}'.format(layer),20)
                img_name = '/code/heatmaps/1/heatmap.{}.model{}.layer{}.jpg'.format(name.split('.')[0], model_name, layer)
                cv2.imwrite(img_name,img)
                print(img_name)
                # except:
                    # print('2nd error in layer: {}'.format(layer))

bulk_heatmap('2.8.6')