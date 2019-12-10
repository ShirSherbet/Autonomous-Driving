# ref: https://www.learnopencv.com/faster-r-cnn-object-detection-with-pytorch/

import cv2
import torchvision
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

VIHICLE_LIST = ['car', 'bus', 'truck']


def get_prediction(img_path, threshold):
    img = Image.open(img_path)

    # Defing PyTorch transform and apply the transform to the image
    transform = T.Compose([T.ToTensor()])
    img = transform(img)

    # Pass the image to the model
    pred = model([img])

    # Get the lables, bouding boxes and prediction score
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())

    # Get list of index with score greater than threshold
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]

    return pred_boxes, pred_class


def object_detection_api(img_path, threshold=0.5, rect_th=2, text_size=1, text_th=2):
    # Get predictions
    boxes, pred_cls = get_prediction(img_path, threshold)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Store only vehicle bounding boxes
    vehicle_box = []

    # Visualize bounding boxes and lables only for vehicles
    for i in range(len(boxes)):
        if pred_cls[i] in VIHICLE_LIST:
            vehicle_box.append(boxes[i])
            cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)
            cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0),
                        thickness=text_th)

    # display the output image
    plt.figure(figsize=(20, 30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

    return vehicle_box


if __name__ == '__main__':
    vb = object_detection_api('./data/train/image_left/um_000011.jpg')
