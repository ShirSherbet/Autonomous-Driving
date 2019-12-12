# Autonomous-Driving

Demo 
===========
See https://github.com/ShirSherbet/Autonomous-Driving/blob/master/report.pdf

Require packages:
===========
- python 3.6.8
- torch 1.3.1
- torchvision 0.4.2
- Keras 2.3.1
- tensorflow 2.0.0
- Pillow 6.1.0
- open3d 0.8.0.0
- matplotlib 3.1.2
- webcolors 1.10
- numpy 1.17.4
- scipy
- scikit-image 0.15.0
- opencv3.4.2

Data folder structure
========
- The KITTI dataset for this project is in data folder
    - form a data folder like this:
       - test:
            - calib
            - image_left
            - image_right
       - train:
            - calib
            - image_left
            - image_right
            - gt_image_left
       - train_angle:
            - image
            - labels

- The pre-trained model for road detection is road_model_final.h5, for road segmentation it is md.pth, for car viewpoint it is model.h5

current scripts for each subtasks:
==========
- get disparity map and depth map

    ```
    python image_processing.py
    ```

    The line 114 is our implemented method for disparity map, and line 116 is the openCV API call for diparity map

- get road detection ground truth mask and visualization

    ```
    python road_detection.py
    ```
    If you don't want to train the model yourself, you could comment out line 117-206
    
    Reference: https://www.depends-on-the-definition.com/unet-keras-segmenting-images/
    
- get road segmentation visualization

    ```
    python road_segmentation.py
    ```
    If you don't want to train the model yourself, you could comment out line 298-305
    
    Reference: https://arxiv.org/pdf/1808.04450.pdf

- fit a ground plane and visualize it with 3D Points Cloud

    ```
    python road_visualization.py
    ```

- detect cars in the image

    ```
    python car_detection.py
    ```
    Reference: https://www.learnopencv.com/faster-r-cnn-object-detection-with-pytorch/
    
- get car viewpoint

    ```
    python car_viewpoint.py
    ```
    This file is just for training the model and visualize the loss/accuracy during the training
    
    Reference: https://www.learnopencv.com/image-classification-using-convolutional-neural-networks-in-keras/ 

- visualize cars and their viewpoint in the image

    ```
    python car_visualization.py
    ```
    This file loads pre-trained model

Contributers:
==========
<table>
  <tr>
    <td align="center"><img src="https://avatars.githubusercontent.com/u/1500684?v=3" width="100px;" alt="Fanxuan Guo"/><br /><sub><b>Fanxuan Guo</b></sub></a><br />👀📖</td>
    <td align="center"><img src="https://avatars.githubusercontent.com/u/1500684?v=3" width="100px;" alt="Shiqi Lin"/><br /><sub><b>Fanxuan Guo</b></sub></a><br />📦💡</td>
   </tr>
</table>
