#!/usr/bin/env python
# -*- coding: utf-8 -*-
# (c) Copyright 2019 Enric Moreu. All Rights Reserved.

import time
start = time.time()


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras import callbacks
from keras import optimizers
from keras import regularizers
from utils.telegram import send
from utils.clr import OneCycleLR
from keras.applications import InceptionV3

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

experiment = '3.10.8'

train_path = '/data/resized_299/train'
validation_path = '/data/resized_299/validation'
test_path = '/data/resized_299/test'
epochs = 200
steps_per_epoch = 200
validation_steps=50
batch_size = 64
lr = 1e-5
l2 = 0.005


#Load data + augmentation
train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.1,
       fill_mode='nearest',
        rotation_range=15)

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(299, 299),
        batch_size=batch_size,
        class_mode='binary')

validation_datagen = ImageDataGenerator(
        rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
        validation_path,
        target_size=(299, 299),
        batch_size=batch_size,
        class_mode='binary')

test_datagen = ImageDataGenerator(
        rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(299, 299),
        batch_size=batch_size,
        class_mode='binary')



# Define model
base_model = InceptionV3(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu',activity_regularizer=regularizers.l2(l2))(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu',activity_regularizer=regularizers.l2(l2))(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.summary()
# Define optimizer
opt = optimizers.Adam(lr=lr)

model.compile(loss = 'binary_crossentropy',
              optimizer = opt,
              metrics = ['accuracy'])

## Callbacks

# LR reduce
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=15, min_lr=1e-6, verbose=1)

# Tensorboard
tbCallBack = callbacks.TensorBoard(log_dir='/code/logs/{}'.format(experiment))

# Checkpoints
checkpoints = callbacks.ModelCheckpoint('/code/checkpoints/{}.weights'.format(experiment), monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

# Terminate on NaN
tnan = callbacks.TerminateOnNaN()

## Train model
model.fit_generator(
       train_generator,
       epochs=epochs,
       steps_per_epoch=steps_per_epoch,
       validation_steps=validation_steps,
       validation_data=validation_generator,
       callbacks=[
        tbCallBack,
        checkpoints
        # tnan
        ],
       shuffle=True,
       verbose=1,
       workers=4,
       use_multiprocessing=True)

## Evaluate model

# Load best model
best_model = load_model('/code/checkpoints/{}.weights'.format(experiment))



# Forward test images
results = best_model.evaluate_generator(test_generator,
        workers=4,
        use_multiprocessing=True)

end = time.time()
total_time = (end - start) // 60

send('''Experiment {} finished in {} minutes

LR: {}
Test accuracy: {}
'''.format(experiment, int(total_time), lr, '%.3f'%(results[1]*100)))
