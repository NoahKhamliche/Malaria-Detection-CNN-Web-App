# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 10:37:52 2019

@author: Shadow
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import cv2
from concurrent import futures
import threading
import matplotlib.pyplot as plt
#%matplotlib inline
# encode text category labels
from sklearn.preprocessing import LabelEncoder

import keras
import datetime
from keras.models import load_model

def train_model():
    base_dir = os.path.join('./cell_images')
    infected_dir = os.path.join(base_dir,'Parasitized')
    healthy_dir = os.path.join(base_dir,'Uninfected')

    infected_files = glob.glob(infected_dir+'/*.png')
    healthy_files = glob.glob(healthy_dir+'/*.png')
    print(len(infected_files), len(healthy_files))


    np.random.seed(42)

    files_df = pd.DataFrame({
        'filename': infected_files + healthy_files,
        'label': ['malaria'] * len(infected_files) + ['healthy'] * len(healthy_files)
    }).sample(frac=1, random_state=42).reset_index(drop=True)

    files_df.head()

    train_files, test_files, train_labels, test_labels = train_test_split(files_df['filename'].values,
                                                                          files_df['label'].values,
                                                                          test_size=0.3, random_state=42)
    train_files, val_files, train_labels, val_labels = train_test_split(train_files,
                                                                        train_labels,
                                                                        test_size=0.1, random_state=42)

    print(train_files.shape, val_files.shape, test_files.shape)
    print('Train:', Counter(train_labels), '\nVal:', Counter(val_labels), '\nTest:', Counter(test_labels))


    def get_img_shape_parallel(idx, img, total_imgs):
        if idx % 5000 == 0 or idx == (total_imgs - 1):
            print('{}: working on img num: {}'.format(threading.current_thread().name,
                                                      idx))
        return cv2.imread(img).shape

    ex = futures.ThreadPoolExecutor(max_workers=None)
    data_inp = [(idx, img, len(train_files)) for idx, img in enumerate(train_files)]
    print('Starting Img shape computation:')
    train_img_dims_map = ex.map(get_img_shape_parallel,
                                [record[0] for record in data_inp],
                                [record[1] for record in data_inp],
                                [record[2] for record in data_inp])
    train_img_dims = list(train_img_dims_map)
    print('Min Dimensions:', np.min(train_img_dims, axis=0))
    print('Avg Dimensions:', np.mean(train_img_dims, axis=0))
    print('Median Dimensions:', np.median(train_img_dims, axis=0))
    print('Max Dimensions:', np.max(train_img_dims, axis=0))


    IMG_DIMS = (125, 125)

    def get_img_data_parallel(idx, img, total_imgs):
        if idx % 5000 == 0 or idx == (total_imgs - 1):
            print('{}: working on img num: {}'.format(threading.current_thread().name,
                                                      idx))
        img = cv2.imread(img)
        img = cv2.resize(img, dsize=IMG_DIMS,
                         interpolation=cv2.INTER_CUBIC)
        img = np.array(img, dtype=np.float32)
        return img

    ex = futures.ThreadPoolExecutor(max_workers=None)
    train_data_inp = [(idx, img, len(train_files)) for idx, img in enumerate(train_files)]
    val_data_inp = [(idx, img, len(val_files)) for idx, img in enumerate(val_files)]
    test_data_inp = [(idx, img, len(test_files)) for idx, img in enumerate(test_files)]

    print('Loading Train Images:')
    train_data_map = ex.map(get_img_data_parallel,
                            [record[0] for record in train_data_inp],
                            [record[1] for record in train_data_inp],
                            [record[2] for record in train_data_inp])
    train_data = np.array(list(train_data_map))

    print('\nLoading Validation Images:')
    val_data_map = ex.map(get_img_data_parallel,
                            [record[0] for record in val_data_inp],
                            [record[1] for record in val_data_inp],
                            [record[2] for record in val_data_inp])
    val_data = np.array(list(val_data_map))

    print('\nLoading Test Images:')
    test_data_map = ex.map(get_img_data_parallel,
                            [record[0] for record in test_data_inp],
                            [record[1] for record in test_data_inp],
                            [record[2] for record in test_data_inp])
    test_data = np.array(list(test_data_map))

    print(train_data.shape, val_data.shape, test_data.shape)

    plt.figure(1 , figsize = (8 , 8))
    n = 0
    for i in range(16):
        n += 1
        r = np.random.randint(0 , train_data.shape[0] , 1)
        plt.subplot(4 , 4 , n)
        plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
        plt.imshow(train_data[r[0]]/255.)
        plt.title('{}'.format(train_labels[r[0]]))
        plt.xticks([]) , plt.yticks([])


    BATCH_SIZE = 64
    NUM_CLASSES = 2
    EPOCHS = 25
    INPUT_SHAPE = (125, 125, 3)

    train_imgs_scaled = train_data / 255.
    val_imgs_scaled = val_data / 255.


    le = LabelEncoder()
    le.fit(train_labels)
    train_labels_enc = le.transform(train_labels)
    val_labels_enc = le.transform(val_labels)

    print(train_labels[:6], train_labels_enc[:6])

    inp = keras.layers.Input()

    conv1 = keras.layers.Conv2D(32, kernel_size=(3, 3),
                                   activation='relu', padding='same')(inp)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = keras.layers.Conv2D(64, kernel_size=(3, 3),
                                   activation='relu', padding='same')(pool1)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    inp = keras.layers.Input(shape=INPUT_SHAPE)
    conv3 = keras.layers.Conv2D(128, kernel_size=(3, 3),
                                   activation='relu', padding='same')(pool2)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    flat = keras.layers.Flatten()(pool3)

    hidden1 = keras.layers.Dense(512, activation='relu')(flat)
    drop1 = keras.layers.Dropout(rate=0.3)(hidden1)
    hidden2 = keras.layers.Dense(512, activation='relu')(drop1)
    drop2 = keras.layers.Dropout(rate=0.3)(hidden2)

    out = keras.layers.Dense(1, activation='sigmoid')(drop2)

    model = keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
    model.summary()



    logdir = os.path.join('.\logs',
                          datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=2, min_lr=0.000001)

    #early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
    #                                              mode='auto', baseline=None, restore_best_weights=False)
    callbacks = [reduce_lr, tensorboard_callback]

    history = model.fit(x=train_imgs_scaled, y=train_labels_enc,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(val_imgs_scaled, val_labels_enc),
                        callbacks=callbacks,
                        verbose=1)

    model.save('basic_cnn.h5')
    return model

def load_model():
    model = load_model('basic_cnn.h5')
    return model

if __name__ == "__main__":
    model = train_model()
    model = load_model()