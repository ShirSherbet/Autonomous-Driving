import cv2
import matplotlib.pyplot as plt
import numpy as np
from car_detection import object_detection_api
from keras.engine.saving import load_model
from skimage import io

ANGLE_DICT = {0: (0, -2), 1: (-1, -np.sqrt(3)), 2: (-np.sqrt(3), -1), 3: (-2, 0), 4: (-np.sqrt(3), 1),
              5: (-1, np.sqrt(3)), 6: (0, 2), 7: (1, np.sqrt(3)), 8: (np.sqrt(3), 1), 9: (2, 0),
              10: (np.sqrt(3), -1), 11: (1, -np.sqrt(3))}

ANGLE_LABEL = {0: '0°', 1: '30°', 2: '60°', 3: '90°', 4: '120°', 5: '150°', 6: '180°', 7: '210°', 8: '240°', 9: '270°',
               10: '300°', 11: '330°'}


def visualize_car(img_path):
    # Get car detection result and trained model
    boxes = object_detection_api(img_path)
    model = load_model('model.h5')

    img = io.imread(img_path)
    fig, ax = plt.subplots()
    ax.imshow(img)

    for i in range(len(boxes)):
        # Use bounding boxes to crop patch
        left, top = int(boxes[i][0][0]), int(boxes[i][0][1])
        right, bottom = int(boxes[i][1][0]), int(boxes[i][1][1])

        # Ignore patches that are too small
        width, height = right - left, bottom - top
        if width >= 30 and height >= 30:
            patch = img[top:bottom, left:right]
            patch = cv2.resize(patch, dsize=(97, 53), interpolation=cv2.INTER_CUBIC)
            patch = patch.reshape((1, 97, 53, 3))

            # Predict viewpoint
            prediction = model.predict_classes(patch)
            dx, dy = ANGLE_DICT[prediction[0]][0], ANGLE_DICT[prediction[0]][1]

            # Visualize car bounding box and angle
            center_x, center_y = int((left + right) / 2), int((top + bottom) / 2)
            rect = plt.Rectangle((left, top), width, height, fill=False, linewidth=2, color='lime')
            plt.arrow(center_x, center_y, dx * 20, -dy * 20, color='w', linewidth=3, head_width=20, head_length=4)
            plt.text(center_x, center_y, ANGLE_LABEL[prediction[0]], fontsize=12, color='lime')
            ax.add_patch(rect)

    plt.show()


if __name__ == '__main__':
    visualize_car('./data/train/image_left/um_000011.jpg')
