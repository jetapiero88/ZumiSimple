from zumicloud.zumimlcloudutils import download_dataset_from_cloud, send_model_to_cloud
#from sklearn.metrics import classification_report, confusion_matrix
from skimage import transform
from skimage import exposure
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import random
import pathlib
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense


def build(width, height, depth, classes):
	# initialize the model along with the input shape to be
	# "channels last" and the channels dimension itself
	model = Sequential()
	inputShape = (height, width, depth)
	chanDim = -1

	# CONV => RELU => BN => POOL
	model.add(Conv2D(8, (5, 5), padding="same",
		input_shape=inputShape))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# first set of (CONV => RELU => CONV => RELU) * 2 => POOL
	model.add(Conv2D(16, (3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(Conv2D(16, (3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# second set of (CONV => RELU => CONV => RELU) * 2 => POOL
	model.add(Conv2D(32, (3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(Conv2D(32, (3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# first set of FC => RELU layers
	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation("relu"))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))

	# second set of FC => RELU layers
	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation("relu"))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))

	# softmax classifier
	model.add(Dense(classes))
	model.add(Activation("softmax"))

		# return the constructed network architecture
	return model

def load_images(dataset_directory):
    
    data_path_train = os.getcwd() +'/'+ dataset_directory + '/' + 'train'

    data_dir_train = pathlib.Path(data_path_train)

    train_images = list(data_dir_train.glob('*/*'))
    train_images = [str(path) for path in train_images]
    random.shuffle(train_images)
 
    #label_names = np.array([item.name for item in data_dir_train.glob('*')])
    label_names = [item.name for item in data_dir_train.glob('*') if item.name != '.ipynb_checkpoints']

    label_dict = {name: i for (i,name) in enumerate(label_names)}

    train_labels=[label_dict[pathlib.Path(path).parent.name] for path in train_images]
 
    data_size=len(train_images)

    ### Testing data

    data_path_test = os.getcwd() +'/'+ dataset_directory + '/' + 'test'

    data_dir_test = pathlib.Path(data_path_test)

    test_images = list(data_dir_test.glob('*/*'))
    test_images = [str(path) for path in test_images]
    random.shuffle(test_images)

    test_labels=[label_dict[pathlib.Path(path).parent.name] for path in test_images]
 
    test_data_size=len(test_images)
    
    return train_images, train_labels, test_images, test_labels

def load_split(x, y, IMG_SIZE):
	# initialize the list of data and labels
	data = []
	labels = []

	# loop over the rows of the CSV file
	for (i, imagepath) in enumerate(x):
		# check to see if we should show a status update

		# split the row into components and then grab the class ID
		# and image path
		image = io.imread(imagepath)

		# resize the image to be 32x32 pixels, ignoring aspect ratio,
		# and then perform Contrast Limited Adaptive Histogram
		# Equalization (CLAHE)
		image = transform.resize(image, (IMG_SIZE, IMG_SIZE))
		image = exposure.equalize_adapthist(image, clip_limit=0.1)

		# update the list of data and labels, respectively
		data.append(image)
		labels.append(int(y[i]))

	# convert the data and labels to NumPy arrays
	data = np.array(data)
	labels = np.array(labels)

	# return a tuple of the data and labels
	return (data, labels)


def train_and_upload_model(dataset_name_zip, model_name):

	dataset_directory = dataset_name_zip.split('.')[0]

	downloaded = download_dataset_from_cloud(dataset_name_zip, dataset_directory)

	if downloaded:
		train_images, train_labels, test_images, test_labels = load_images(dataset_directory)
    else:
        return print('There was a problem downloading the dataset, make sure you input the right name')
    
    IMG_SIZE=32

    (x_train, y_train) = load_split(train_images, train_labels, IMG_SIZE)
    (x_test, y_test) = load_split(test_images, test_labels, IMG_SIZE)

    # scale data to the range of [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
 
    # one-hot encode the training and testing labels
    numLabels  = len(label_names)
    y_train = to_categorical(y_train, numLabels)
    y_test = to_categorical(y_test, numLabels)
 
    # account for skew in the labeled data
    classTotals = y_train.sum(axis=0)
    classWeight = classTotals.max() / classTotals

    model = build(width=32, height=32, depth=3,
        classes=numLabels)

    NUM_EPOCHS = 20
    INIT_LR = 1e-3
    BS = 10

    opt = Adam(lr=INIT_LR, decay=INIT_LR / (NUM_EPOCHS * 0.5))

    model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])

    aug = ImageDataGenerator(
        rotation_range=5,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.25,
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode="nearest")
    
    H = model.fit_generator(
        aug.flow(x_train, y_train, batch_size=BS),
        validation_data=(x_test, y_test),
        steps_per_epoch=100,
        epochs=NUM_EPOCHS,
        class_weight=classWeight,
        verbose=1)

    model_path = os.getcwd() + '/' + model_name

    os.mkdir(model_path)

    model.save_weights(model_path + '/' + model_name + 'weights.h5')
    model_json = model.to_json()
    with open(model_path + '/' + model_name +'.json',"w") as json_file:
        json_file.write(model_json)
    json_file.close()
    
    uploaded = send_model_to_cloud(model_name)

    if uploaded:
        return print('Your model was trained and uploaded to the cloud as: {}'.format(model_name + '.zip'))
