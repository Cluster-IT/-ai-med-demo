#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(u'pip install imutils')


# In[4]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
#from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shutil
import random
import cv2
import os


# In[ ]:


dataset_path = 'C:/Users/ADMIN/Desktop/Covid 19_Large'


# In[ ]:


covid_dataset_path = 'C:/Users/ADMIN/Desktop/Covid 19_Large/covid'


# In[ ]:


pneumonia_dataset_path = 'C:/Users/ADMIN/Desktop/Covid 19_Large/pneumonia'


# In[ ]:


normal_dataset_path = 'C:/Users/ADMIN/Desktop/Covid 19_Large/normal'


# In[ ]:


def ceildiv(a, b):
    return -(-a // b)

def plots_from_files(imspaths, figsize=(20,10), rows=1, titles=None, maintitle=None):
    """Plot the images in a grid"""
    f = plt.figure(figsize=figsize)
    if maintitle is not None: plt.suptitle(maintitle, fontsize=10)
    for i in range(len(imspaths)):
        sp = f.add_subplot(rows, ceildiv(len(imspaths), rows), i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        img = plt.imread(imspaths[i])
        plt.imshow(img)


# In[ ]:


covid_images = list(paths.list_images(f"{dataset_path}/covid"))
pneumonia_images = list(paths.list_images(f"{dataset_path}/pneumonia"))
normal_images = list(paths.list_images(f"{dataset_path}/normal"))


# In[ ]:


plots_from_files(covid_images, rows=30, maintitle="Covid X-ray images")


# In[ ]:


plots_from_files(pneumonia_images, rows=25, maintitle="Pneumonia X-ray images")


# In[ ]:


plots_from_files(normal_images, rows=25, maintitle="Normal X-ray images")


# # Data preprocessing

# In[ ]:


# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 4e-3
EPOCHS = 8
BS = 20


# In[ ]:


# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(dataset_path))
data = []
labels = []
# loop over the image paths
for imagePath in imagePaths:
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]
    # load the image, swap color channels, and resize it to be a fixed
    # 224x224 pixels while ignoring aspect ratio
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)
# convert the data and labels to NumPy arrays while scaling the pixel
# intensities to the range [0, 1]
data = np.array(data) / 255.0
labels = np.array(labels)


# ### Perform one-hot encoding on the labels

# In[ ]:


# perform one-hot encoding on the labels
lb = MultiLabelBinarizer()


# In[ ]:


labels = LabelEncoder().fit_transform(labels)


# In[ ]:


labels = to_categorical(labels)
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)
# initialize the training data augmentation object
trainAug = ImageDataGenerator(rotation_range=15, fill_mode="nearest")


# In[ ]:


# load the VGG16 network, ensuring the head FC layer sets are left
# off
baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(3, activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False


# In[ ]:


# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")
H = model.fit_generator(
    trainAug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)


# In[ ]:




