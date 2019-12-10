# ref: https://www.depends-on-the-definition.com/unet-keras-segmenting-images/

import os
import cv2
import numpy as np
from keras import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.engine.saving import load_model
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import webcolors

im_width = 512
im_height = 128
split = 0.8

callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint('road_model_checkpoint.h5', verbose=1, save_best_only=True, save_weights_only=True)
]


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path -- encoder
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout * 0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # expansive path -- decoder
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


def overlay_mask(image, mask):
    # pre-process mask
    mask[np.where(mask < 0.5)] = 0  # not road pixel
    mask[np.where(mask >= 0.5)] = 1 # road pixel
    mask = mask.astype(int)

    # code from tutorial7 to visualize segmentation
    obj_ids = np.unique(mask)
    number_object = obj_ids.shape[0]

    count = 0
    for o_id in obj_ids:
        mask[mask == o_id] = count
        count += 1

    base_COLORS = []

    for key, value in mcolors.CSS4_COLORS.items():
        rgb = webcolors.hex_to_rgb(value)
        base_COLORS.append([rgb.blue, rgb.green, rgb.red])
    base_COLORS = np.array(base_COLORS)

    np.random.seed(99)
    base_COLORS = np.random.permutation(base_COLORS)

    colour_id = np.array([(id) % len(base_COLORS) for id in range(number_object)])
    COLORS = base_COLORS[colour_id]
    COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")
    mask = COLORS[mask]
    output = ((0.4 * image) + (0.6 * mask)).astype("uint8")
    io.imshow(output)
    io.show()


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


def visualize_predict(model_name):
    model = load_model(model_name)

    images = []
    images_path = []
    images_shape = []
    for path in os.listdir('data/test/image_left'):
        img_full_path = 'data/test/image_left' + '/' + path
        if "DS_Store" not in img_full_path:
            img = io.imread(img_full_path, as_gray=True)
            images_shape.append(img.shape)
            img = cv2.resize(img, dsize=(im_width, im_height), interpolation=cv2.INTER_CUBIC)
            img = img.reshape((im_height, im_width, 1))
            images.append(img)
            images_path.append(img_full_path)

    prediction = model.predict(np.array(images), verbose=1)

    for i in range(len(images)):
        pred = prediction[i][:, :, 0]
        pred = cv2.resize(pred, dsize=(images_shape[i][1], images_shape[i][0]), interpolation=cv2.INTER_CUBIC)
        io.imshow(pred)
        io.show()
        img = io.imread(images_path[i])
        overlay_mask(img, pred)


if __name__ == '__main__':
    input_img = Input((im_height, im_width, 1), name='img')
    model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)

    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

    # Read images
    images = []
    masks = []
    for path in os.listdir('data/train/image_left'):
        img_full_path = 'data/train/image_left' + '/' + path
        if "DS_Store" not in img_full_path:
            img = io.imread(img_full_path, as_gray=True)
            img = cv2.resize(img, dsize=(im_width, im_height), interpolation=cv2.INTER_CUBIC)
            img = img.reshape((im_height, im_width, 1))
            images.append(img)

            num = path.split('.')[0].split('_')[1]
            mask_full_path = 'data/train/gt_image_left' + '/um_road_' + num + '.png'
            mask = io.imread(mask_full_path, as_gray=True)
            mask = cv2.resize(mask, dsize=(im_width, im_height), interpolation=cv2.INTER_CUBIC)
            mask = mask.reshape((im_height, im_width, 1))
            masks.append(mask)

    divisor = int(len(images) * split)

    train_data, train_label = np.array(images[:divisor]), np.array(masks[:divisor])
    test_data, test_label = np.array(images[divisor:]), np.array(masks[divisor:])

    results = model.fit(train_data, train_label, batch_size=15, epochs=40, callbacks=callbacks,
                        validation_data=(test_data, test_label))
    model.save("road_model_final.h5")

    # Visualize model performance
    visualize_performance(results)

    # Visualize predict
    visualize_predict('road_model_final.h5')
