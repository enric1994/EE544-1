#!/usr/bin/env python
# -*- coding: utf-8 -*-
# (c) Copyright 2019 Enric Moreu. All Rights Reserved.

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten
from keras import callbacks
from keras import optimizers
from keras.applications import InceptionV3

#Remove warnings????
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

#Not working with pretrained model
# import keras.backend as K
# K.set_floatx('float16')

experiment = '3.1.14'

train_path = '/data/resized_299/train'
validation_path = '/data/resized_299/validation'
image_size = 299
epochs = 500
batch_size = 32
steps_per_epoch = 1600
validation_steps = 400

#Load data + augmentation
train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
        # shear_range=0.2,
        brightness_range=(0.5,1.5),
        # zca_whitening=True,
        # featurewise_std_normalization=True,
        horizontal_flip=True,
        rotation_range=15,
        width_shift_range=.15,
        height_shift_range=.15)

val_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
        # shear_range=0.2,
        brightness_range=(0.5,1.5),
        # zca_whitening=True,
        # featurewise_std_normalization=True,
        horizontal_flip=True,
        rotation_range=15,
        width_shift_range=.15,
        height_shift_range=.15)

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary') 

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = val_datagen.flow_from_directory(
        validation_path,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary') 


# Define model
base_model = InceptionV3(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
# x = Dropout(0.5)(x)
# x = Flatten(input_shape=(2048,))(x)
# x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True
# print(model.summary())

# Define optimizer
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss = 'binary_crossentropy',
              optimizer = sgd,
              metrics = ['accuracy'])

# Tensorboard
tbCallBack = callbacks.TensorBoard(log_dir='/code/logs/{}'.format(experiment))

# LR scheduler
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=10, min_lr=0.0000000001, verbose=1)

model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch // batch_size, #total images/bs (modify names)
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps // batch_size, #total images/bs
        callbacks=[tbCallBack, reduce_lr],
        shuffle=True)