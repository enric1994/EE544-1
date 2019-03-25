#!/usr/bin/env python
# -*- coding: utf-8 -*-
# (c) Copyright 2019 Enric Moreu. All Rights Reserved.

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras import callbacks
from keras import optimizers


# import keras.backend as K
# K.set_floatx('float16')

experiment = '1.2.8'

train_path = '/data/resized_224/train'
validation_path = '/data/resized_224/validation'
epochs = 100
batch_size = 32
steps_per_epoch = 1600
validation_steps = 384

#Load data + augmentation
train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=15,
        width_shift_range=.15,
        height_shift_range=.15)

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary') 

validation_datagen = ImageDataGenerator(
        rescale=1./255)

validation_generator = train_datagen.flow_from_directory(
        validation_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary') 


# Define model
model = Sequential()
model.add(Conv2D(128, (3, 3), input_shape=(224,224,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Define optimizer
opt = optimizers.SGD(lr=1e-1, decay=0, momentum=0.9, nesterov=False)
# opt = optimizers.Adam(lr=1e-2)

model.compile(loss = 'binary_crossentropy',
              optimizer = opt,
              metrics = ['accuracy'])

# LR reduce
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=10, min_lr=0.0000000001, verbose=2)

# Tensorboard
tbCallBack = callbacks.TensorBoard(log_dir='/code/logs/{}'.format(experiment))

# Checkpoints
checkpoints = callbacks.ModelCheckpoint('/code/checkpoints/{}.weights'.format(experiment), monitor='val_acc', verbose=2, save_best_only=True, save_weights_only=False, mode='auto', period=1)


from utils.clr import LRFinder
lr_finder = LRFinder(1600, batch_size, minimum_lr=1e-6, maximum_lr=10.,
                     lr_scale='exp',
                     save_dir='/code/docs/plots/lr_finder', verbose=True)
epochs = 1

model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps // batch_size,
        callbacks=[tbCallBack, checkpoints, lr_finder])

lr_finder.plot_schedule_from_file('/code/docs/plots/lr_finder',experiment)
