#!/usr/bin/env python
# -*- coding: utf-8 -*-
# (c) Copyright 2019 Enric Moreu. All Rights Reserved.

import time
start = time.time()

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras import callbacks
from keras import optimizers
from utils.telegram import send

experiment = '2.9.0.final'

train_path = '/data/resized_224/train'
validation_path = '/data/resized_224/validation'
test_path = '/data/resized_224/test'
epochs = 50
steps_per_epoch = 300
validation_steps=50
batch_size = 32
lr=5e-6
decay=0

## Load data + augmentation
train_datagen = ImageDataGenerator(
        rescale=1./255,
       zoom_range=0.2,
        rotation_range=20)


train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary') 

validation_datagen = ImageDataGenerator(
        rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
        validation_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary')

test_datagen = ImageDataGenerator(
        rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary')

## Define model

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(224,224,3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
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

model.add(Flatten())

model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

## Define optimizer
opt = optimizers.Adam(lr=lr,decay=decay)

model.compile(loss = 'binary_crossentropy',
              optimizer = opt,
              metrics = ['accuracy'])

## Callbacks

# Tensorboard
tbCallBack = callbacks.TensorBoard(log_dir='/code/logs/{}'.format(experiment))

# Checkpoints
checkpoints = callbacks.ModelCheckpoint('/code/checkpoints/{}.weights'.format(experiment), monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

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
total_time = (end - start)

send('''Experiment {} finished in {} seconds
LR: {}
Test accuracy: {}
'''.format(experiment, int(total_time), lr, '%.2f'%(results[1]*100)))