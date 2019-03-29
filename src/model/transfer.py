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

experiment = '3.3.1'

train_path = '/data/resized_299/train'
validation_path = '/data/resized_299/validation'
test_path = '/data/resized_299/test'
epochs = 50
batch_size = 64
lr = 1e-3
decay = 1e-4
max_lr=1e-1
l1 = 0.005
l2 = 0.005
fine_model = '3.2.8'


#Load data + augmentation
train_datagen = ImageDataGenerator(
        rescale=1./255,
       zoom_range=0.3,
       fill_mode='nearest',
#        samplewise_center=True,
#        samplewise_std_normalization=True,
        rotation_range=40,
       width_shift_range=0.3,
       height_shift_range=0.3,
       horizontal_flip=True,
       vertical_flip=True)

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
# base_model = InceptionV3(weights='imagenet', include_top=False)
base_model = load_model('/code/checkpoints/{}.weights'.format(fine_model))

# Remove last 2 layers
base_model.layers.pop()
base_model.layers.pop()


x = base_model.output
# x = GlobalAveragePooling2D()(x)
x = Dense(512, name='dense_2', activation='relu')(x)
# x = BatchNormalization(name='batch_normalization_XX')(x)
# x = Activation('relu')(x)
x = Dropout(0.2, name='dropout_2')(x)
x = Dense(128, name='dense_3', activation='relu')(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
x = Dropout(0.2, name='dropout_3')(x)
predictions = Dense(1, name='dense_4', activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in model.layers[:-10]:
   layer.trainable = False
for layer in model.layers[-10:]:
   layer.trainable = True
#    if isinstance(layer, Conv2D):
#         layer.add_loss(regularizers.l1_l2(l1=l1, l2=l2)(layer.kernel))

# for layer in model.layers:
#     if isinstance(layer, Conv2D):
#         layer.add_loss(regularizers.l2(alpha)(layer.kernel))

# Define optimizer
opt = optimizers.Adam(lr=lr, decay=decay)

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

# One Cycle
lr_manager = OneCycleLR(max_lr, batch_size, 1600, scale_percentage=0.1,
                        maximum_momentum=0, minimum_momentum=0, verbose=True)
# Terminate on NaN
tnan = callbacks.TerminateOnNaN()

## Train model
model.fit_generator(
       train_generator,
       epochs=epochs,
       validation_data=validation_generator,
       callbacks=[
        tbCallBack,
        checkpoints,
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
