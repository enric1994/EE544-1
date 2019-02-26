#!/usr/bin/env python
# -*- coding: utf-8 -*-
# (c) Copyright 2019 Enric Moreu. All Rights Reserved.

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras import callbacks
from keras import optimizers


import keras.backend as K
K.set_floatx('float16')

experiment = '2.3.3'

train_path = '/data/resized_224/train'
validation_path = '/data/resized_224/validation'
epochs = 250
batch_size = 32
steps_per_epoch = 2000
validation_steps = 800

#Load data + augmentation
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

val_datagen = ImageDataGenerator(
        rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary') 

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = val_datagen.flow_from_directory(
        validation_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary') 


# Define model

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(224,224,3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), input_shape=(224,224,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Define optimizer
sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

# LR scheduler
reduce_lr = callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.3,
                              patience=5, min_lr=0.00000001)

model.compile(loss = 'binary_crossentropy',
              optimizer = 'sgd',
              metrics = ['accuracy'])

# Tensorboard
tbCallBack = callbacks.TensorBoard(log_dir='/code/logs/{}'.format(experiment))

model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps // batch_size,
        callbacks=[tbCallBack, reduce_lr])