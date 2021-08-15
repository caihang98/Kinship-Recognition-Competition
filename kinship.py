# -*- coding: utf-8 -*-


from google.colab import drive
drive.mount('/gdrive')

"""Install Libraries"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install keras_vggface
# !pip install keras_applications

from collections import defaultdict
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from glob import glob
from random import choice, sample
from keras import backend as K
from keras.layers.core import Lambda
import tensorflow as tf
import random
import keras
import numpy as np
from PIL import Image as im
import cv2
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import cosine
import numpy as np
import pandas as pd
from tensorflow.keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, Flatten, Input, Dense, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract, Reshape, Average, Add
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace

train_file_path = "/gdrive/MyDrive/Kinship Recognition Starter/train_ds.csv"
train_folders_path = "/gdrive/MyDrive/Kinship Recognition Starter/train/train-faces/"

val_famillies = "F09"

all_images = glob(train_folders_path + "*/*/*.jpg")

print('********************************************************')
print(all_images)


datagen = ImageDataGenerator(

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest')

for i in all_images:
    address = i
    img = load_img(address)
    image = img_to_array(img)
    image = image.reshape((1,)+image.shape)
    j = 0
    for i in datagen.flow(image,batch_size=1,save_prefix="picture",save_format='jpg'):
        j+=1
        if(j>=7):
            break;
print("done")






train_images = [x for x in all_images if val_famillies not in x]
val_images = [x for x in all_images if val_famillies in x]

train_person_to_images_map = defaultdict(list)

ppl = [x.split("/")[-3] + "/" + x.split("/")[-2] for x in all_images]

for x in train_images:
    train_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

val_person_to_images_map = defaultdict(list)

for x in val_images:
    val_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

relationships = pd.read_csv(train_file_path)
relationships = list(zip(relationships.p1.values, relationships.p2.values, relationships.relationship.values))
relationships = [(x[0],x[1],x[2]) for x in relationships if x[0][:10] in ppl and x[1][:10] in ppl]

train = [x for x in relationships if val_famillies not in x[0]]
val = [x for x in relationships if val_famillies in x[0]]


from keras.preprocessing import image


def read_img(path):
    img = image.load_img(path, target_size=(224, 224))
    img = np.array(img).astype(np.float)
    return preprocess_input(img, version=2)


def gen(list_tuples, person_to_images_map, batch_size=16):
    ppl = list(person_to_images_map.keys())
    while True:
        batch_tuples = sample(list_tuples, batch_size)
        
        # All the samples are taken from train_ds.csv, labels are in the labels column
        labels = []
        for tup in batch_tuples:
          labels.append(tup[2])

        X1 = [x[0] for x in batch_tuples]
        X1 = np.array([read_img(train_folders_path + x) for x in X1])

        X2 = [x[1] for x in batch_tuples]
        X2 = np.array([read_img(train_folders_path + x) for x in X2])

        yield [X1, X2], np.array(labels)


def baseline_model():
    input_1 = Input(shape=(224, 224, 3))
    input_2 = Input(shape=(224, 224, 3))


    base_model = VGGFace(model='resnet50', include_top=False)
   
    for x in base_model.layers[:-3]:
        x.trainable = True

    x1 = base_model(input_1)
    x2 = base_model(input_2)

    #x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(x1), GlobalAvgPool2D()(x1)])
    #x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(x2), GlobalAvgPool2D()(x2)])
    x1 = Dense(100, activation="relu",kernel_regularizer=regularizers.l2(0.001))(x1)
    x2 = Dense(100, activation="relu",kernel_regularizer=regularizers.l2(0.001))(x2)
    


    sub = Conv2D(100 , [1,1] )(Subtract()([x1,x2]))
    mul1 = Conv2D(100 , [1,1] )(Multiply()([x1,x1]))
    mul2 = Conv2D(100 , [1,1] )(Multiply()([x2,x2]))


    x3 = Conv2D(100 , [1,1] )(Multiply()([sub, sub]))
    x4 = Conv2D(100 , [1,1] )(Subtract()([mul1, mul2]))

    x = Concatenate(axis=-1)([x3, x4])

    x = BatchNormalization(axis = -1)(x)
    x = Flatten()(x)

    x = Dense(100, activation="relu")(x)

    x = Dropout(0.5)(x)

    x = Dense(25, activation="relu",kernel_regularizer=regularizers.l1(0.001))(x)
    
    x = Dropout(0.5)(x)

    out = Dense(1, activation="sigmoid")(x)

    model = Model([input_1, input_2], out)

    model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(0.00001))

    model.summary()

    return model

file_path = "/gdrive/MyDrive/vgg_face.h5"

checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')

reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=20, verbose=1)

callbacks_list = [reduce_on_plateau,checkpoint]

model = baseline_model()

model.fit(gen(train, train_person_to_images_map, batch_size=16), use_multiprocessing=True,
                validation_data=gen(val, val_person_to_images_map, batch_size=16), epochs=50, verbose=2,
                workers=4, callbacks=callbacks_list, steps_per_epoch=100, validation_steps=50)


test_path = "/gdrive/MyDrive/Kinship Recognition Starter/test/"

model = baseline_model()
model.load_weights("/gdrive/MyDrive/vgg_face.h5")

submission = pd.read_csv('/gdrive/MyDrive/Kinship Recognition Starter/test_ds.csv')
predictions = []


for i in range(0, len(submission.p1.values), 32):
    X1 = submission.p1.values[i:i+32]
    X1 = np.array([read_img(test_path + x) for x in X1])

    X2 = submission.p2.values[i:i+32]
    X2 = np.array([read_img(test_path + x) for x in X2])

    pred = model.predict([X1, X2]).ravel().tolist()
    predictions += pred

print(predictions)


import numpy as np
thres = np.mean(predictions)+0.1*np.std(predictions)
for i in range(len(predictions)):
  if predictions[i] >= thres:
    predictions[i] = 1
  else:
    predictions[i] = 0
print(predictions)

d = {'index': np.arange(0, 3000, 1), 'label':predictions}

submissionfile = pd.DataFrame(data=d)

print(predictions)

submissionfile.to_csv("/gdrive/MyDrive/predictions.csv", index=False)

