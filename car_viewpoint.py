# ref: https://www.learnopencv.com/image-classification-using-convolutional-neural-networks-in-keras/

import os
import cv2
import numpy as np
from skimage import io
import scipy.io as sio
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization, Activation, GaussianNoise

BATCH_SIZE = 256
EPOCHS = 40
VALIDATION_SPLIT = 0.8
NUM_CLASSES = 12


def read_files(image_path, label_path):
    # Read images and labels
    images = []
    annotations = []
    for path in os.listdir(image_path):
        img_full_path = image_path + '/' + path
        if "DS_Store" not in img_full_path:
            img = io.imread(img_full_path)
            images.append(img)

            num = path.split('.')[0]
            label_full_path = label_path + '/' + num + '.mat'
            anno = sio.loadmat(label_full_path)['annotation'][0][0]
            annotations.append(anno)

    return images, annotations


def get_labels(orient):
    # Use one hot encoder to represent labels
    divisor = int(orient // 30)
    label = [1 if i == divisor else 0 for i in range(12)]
    return np.array(label)


def filter_cars(annotations, img):
    classes = annotations[0][0]
    bboxes = annotations[3]
    orient = annotations[7]
    truncated = annotations[4][0]
    occluded = annotations[8][0]

    # Check if each car is good for training, and return bounding boxes and orientations
    patches, orients = [], []
    height, width = [], []
    for i in range(len(classes)):
        cla = classes[i][0]
        trunc = truncated[i][0]
        occ = occluded[i][0]
        if cla is not 'DontCare' and trunc <= 0.3 and occ <= 2:
            left, top = int(bboxes[i][0]), int(bboxes[i][1])
            right, bottom = int(left + bboxes[i][2]), int(top + bboxes[i][3])
            patch = img[top:bottom, left:right]
            patches.append(patch)
            orients.append(get_labels(orient[i][0]))
            width.append(bboxes[i][2])
            height.append(bboxes[i][3])

    return height, width, patches, orients


def load_data(image_path, label_path, split):
    # Read in images and labels
    images, annotations = read_files(image_path, label_path)

    # Store patch of car and its viewpoint
    patches_list, labels_list = [], []
    heights, widths = [], []
    for i in range(len(images)):
        height, width, patches, orients = filter_cars(annotations[i], images[i])
        patches_list.extend(patches)
        labels_list.extend(orients)
        heights.extend(height)
        widths.extend(width)

    # Calculate average patch width and height
    avg_height = int(sum(heights) / len(heights))
    avg_width = int(sum(widths) / len(widths))

    # Resize patches to have same size
    patches_list = [cv2.resize(patch, dsize=(avg_height, avg_width), interpolation=cv2.INTER_CUBIC) for patch in
                    patches_list]

    # Divide the data into train and test sets with split
    divider = int(len(patches_list) * split)
    train_data, train_labels = np.array(patches_list[:divider]), np.array(labels_list[:divider])
    test_data, test_labels = np.array(patches_list[divider:]), np.array(labels_list[divider:])

    return avg_height, avg_width, train_data, train_labels, test_data, test_labels


def createModel(input_shape, nclasses):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nclasses, activation='softmax'))
    return model


def train_model(image_path, label_path):
    # Load the angle data and split it to train and test sets
    avg_height, avg_width, train_data, train_labels, test_data, test_labels = load_data(image_path, label_path,
                                                                                        VALIDATION_SPLIT)
    # Create a model
    input_shape = (avg_width, avg_height, 3)
    model = createModel(input_shape, NUM_CLASSES)

    # Compile and fit the model, evaluate the model with test data
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(train_data, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,
                        validation_data=(test_data, test_labels))
    model.evaluate(test_data, test_labels)

    # Save model
    model.save("model.h5")

    return history


def visualize_performance(history):
    # Loss Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['loss'], 'r', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)
    plt.show()

    # Accuracy Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['accuracy'], 'r', linewidth=3.0)
    plt.plot(history.history['val_accuracy'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)
    plt.show()


if __name__ == '__main__':
    history = train_model('data/train_angle/image', 'data/train_angle/labels')
    visualize_performance(history)
