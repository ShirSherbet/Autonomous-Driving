# ref: https://arxiv.org/pdf/1808.04450.pdf
# Method 1 for Q3
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import webcolors
import torch
import torch.optim as optim
import torch.nn as nn
import sys
import os
import torch.utils as tu
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import PIL.ImageDraw as ImageDraw

# dataloader
class KittiDataset(tu.data.Dataset):
    def __init__(self, image_path, label_path, transform):
        self.imgs, self.labels = read_files(image_path, label_path)
        self.transform = transform

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        img = self.transform(img)

        dt = {'img': img,
              'gt': torch.IntTensor(label)}

        return dt

    def __len__(self):
        return len(self.imgs)

def read_files(image_path, label_path):
    # read image and pre-process them for the training data loader

    images = []
    labels = []

    for img_name in os.listdir(image_path):
        path = os.path.join(image_path, img_name)
        if "DS_Store" not in path:

            # process image with pyramid prediction scheme
            img = Image.open(path).convert('RGB')
            # crop image at center
            c_img = crop_image(img)
            # rescale image to the same size
            img = ImageOps.fit(img, (600, 160), Image.ANTIALIAS)
            # add x, y coordinate channels
            img = generate_coord_channels(img)
            c_img = generate_coord_channels(c_img)
            images.append(img)
            images.append(c_img)

            # process gt_mask with pyramid prediction scheme
            # get gt_mask name
            img_name = img_name.split('.')[0]
            img_name_l = img_name.split('_')
            img_name_l.insert(1, "road")
            gt_name = '_'.join(img_name_l)
            gt_name += '.png'

            gt_path = os.path.join(label_path, gt_name)
            gt_mask = Image.open(path).convert('L')

            c_mask = crop_image(gt_mask)
            gt_mask = ImageOps.fit(gt_mask, (600, 160), Image.ANTIALIAS)

            # generate ground truth road segmentation boundary vector
            # vector size = 600 * 1, v[i] = the row index of the boundary pixel
            gt_mask = np.array(gt_mask)
            bound = gt_mask.argmax(axis=0)
            gt_col_sum = np.sum(gt_mask, axis=0)
            bound[gt_col_sum == 0] = 160

            c_mask = np.array(c_mask)
            c_bound = c_mask.argmax(axis=0)
            c_col_sum = np.sum(c_mask, axis=0)
            c_bound[c_col_sum == 0] = 160

            labels.append(bound)
            labels.append(c_bound)

    return images, labels

def make_coordinates_matrix(im_shape, step=1):
    """
    Return a matrix of size (im_shape[0] x im_shape[1] x 2) such that g(y,x)=[y,x]
    """
    range_x = np.arange(0, im_shape[1], step)
    range_y = np.arange(0, im_shape[0], step)
    axis_x = np.repeat(range_x[np.newaxis, ...], len(range_y), axis=0)
    axis_y = np.repeat(range_y[..., np.newaxis], len(range_x), axis=1)

    return np.dstack((axis_y, axis_x))

def generate_coord_channels(image):
    np_img = np.array(image)
    coord_m = make_coordinates_matrix(np_img.shape)
    np_img = np.concatenate((np_img, coord_m), axis=2)
    return np_img

def crop_image(image):
    center_x = image.size[0] / 2
    center_y = image.size[1] / 2
    left = center_x - 300
    top = center_y - 80
    right = center_x + 300
    bottom = center_y + 80
    c_img = image.crop((left, top, right, bottom))
    return c_img

# training network
class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.conv_in = nn.Conv2d(in_channels=5, out_channels=64, stride=1, kernel_size=3, padding=1)  # 2
        self.conv_en = nn.Conv2d(in_channels=64, out_channels=64, stride=1, kernel_size=3, padding=1)  # 2
        self.pool = nn.MaxPool2d(2, 2)

        self.conv_fp = nn.Conv2d(in_channels=64, out_channels=64, stride=1, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64)

        self.conv_de1 = nn.Conv2d(in_channels=64, out_channels=64, stride=(5, 1), kernel_size=(5, 1))
        self.conv_de2 = nn.Conv2d(in_channels=64, out_channels=1, stride=1, kernel_size=(4, 1))
        self.upsample = nn.Upsample(scale_factor=(1, 8), mode='nearest')

    def forward(self, image):
        # print("input shape: expect shape 5 * 160 * 600 ",image.shape)
        feature = self.conv_in(image)
        # print("conv1: expect shape 64 * 160 * 600 ",feature.shape)
        feature = self.conv_en(feature)
        # print("conv2: expect shape 64 * 160 * 600 ",feature.shape)
        feature = self.pool(feature)
        # print("pool1: expect shape 64 * 80 * 300 ",feature.shape)

        feature = self.conv_en(feature)
        # print("conv3: expect shape 64 * 80 * 300 ",feature.shape)
        feature = self.conv_en(feature)
        # print("conv4: expect shape 64 * 80 * 300 ",feature.shape)
        feature = self.pool(feature)
        # print("pool2: expect shape 64 * 40 * 150 ",feature.shape)

        feature = self.conv_en(feature)
        # print("conv5: expect shape 64 * 40 * 150 ",feature.shape)
        feature = self.conv_en(feature)
        # print("conv6: expect shape 64 * 40 * 150 ",feature.shape)
        feature = self.pool(feature)
        # print("pool3: expect shape 64 * 20 * 75 ",feature.shape)

        output = self.conv_fp(feature)
        # print("conv7: expect shape 64 * 20 * 75 ", output.shape)
        output = output.view(output.shape[2], output.shape[3], 64)
        hidden_state = torch.randn(1, output.shape[1], 64)
        cell_state = torch.randn(1, output.shape[1], 64)
        output = self.lstm(output, (hidden_state, cell_state))[0]
        output = output.view(1, 64, output.shape[0], output.shape[1])
        # print("lstm1: expect shape 64 * 20 * 75 ", output.shape)

        output = self.conv_fp(output)
        # print("conv8: expect shape 64 * 20 * 75 ", output.shape)
        output = output.view(output.shape[2], output.shape[3], 64)
        hidden_state = torch.randn(1, output.shape[1], 64)
        cell_state = torch.randn(1, output.shape[1], 64)
        output = self.lstm(output, (hidden_state, cell_state))[0]
        output = output.view(1, 64, output.shape[0], output.shape[1])
        # print("lstm2: expect shape 64 * 20 * 75 ", output.shape)

        output = self.conv_de1(output)
        # print("conv9: expect shape 64 * 4 * 75 ", output.shape)
        output = self.conv_de2(output)
        # print("conv10: expect shape 1 * 1 * 75 ", output.shape)
        output = self.upsample(output)
        # print("final output: expect shape 1 * 1 * 600 ", output.shape)
        return output

def train_road_classifier(md, optimizer, trainloader, criterion):
    num_epochs = 80
    total_step = 270
    losses = list()

    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # zero the gradients
            md.zero_grad()
            optimizer.zero_grad()

            # set model into train mode
            md.train()

            images = data['img']
            gt_boundaries = data['gt']

            # Pass the inputs through the CNN-RNN model.
            outputs = md(images.float())

            # Calculate the batch loss
            loss = criterion(outputs[0, 0, :, :], gt_boundaries.float())

            # Backward pass
            loss.backward()

            # Update the parameters in the optimizer
            optimizer.step()

            losses.append(loss.item())
            running_loss += loss.item()

            # save the losses
            np.save('losses', np.array(losses))

            # Get training statistics.
            stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % (
                epoch, num_epochs, i, total_step, loss.item())

            # Print training statistics (on same line).
            print('\r' + stats, end="")
            sys.stdout.flush()

    # Save the weights.
    print("\nSaving the model")
    torch.save(md.state_dict(), os.path.join('./data/train/results-dets', 'md.pth'))

# testing visualization
def visualize_segmentation(boundary):
    # visualize the segmentation mask for the image given the boudnary vector
    image = Image.new("RGB", (600, 160))
    draw = ImageDraw.Draw(image)
    points = generate_boundary_coord(boundary)
    points = np.vstack((points, points[0, :]))
    points = tuple(map(tuple, points))
    draw.polygon((points), fill=(255, 255, 255))
    # image.show()
    return image

def generate_boundary_coord(y, step=1):
    # generate the road boundary xy coordinate based on the boundary row index vector
    im_shape = (1, y.shape[1])

    range_x = np.arange(0, im_shape[1], step)
    range_y = np.arange(0, im_shape[0], step)
    axis_x = np.repeat(range_x[np.newaxis, ...], len(range_y), axis=0)
    result = np.dstack((axis_x, y)).reshape((im_shape[1], 2))

    return result

def reassemble(scaled_img, cropped_img, img_size):
    # reassemble the overlayed scaled and cropped images
    center_x = img_size[1] // 2
    center_y = img_size[0] // 2
    # scale the image back to its original size
    scaled_img = ImageOps.fit(scaled_img, img_size, Image.ANTIALIAS)
    # replaced the pixel values with the cropped image at the center
    scaled_img.paste(cropped_img, (center_x, center_y))
    # scaled_img.show()
    return scaled_img

def overlay_mask(image, gt_mask):
    gt_mask = gt_mask.astype(int)
    # code from tutorial7 to visualize segmentation
    obj_ids = np.unique(gt_mask)
    number_object = obj_ids.shape[0]


    count = 0
    for o_id in obj_ids:
        gt_mask[gt_mask == o_id] = count
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
    mask = COLORS[gt_mask]
    output = ((0.4 * image) + (0.6 * mask)).astype("uint8")
    plt.imshow(output[:, :, ::-1])
    plt.show()

if __name__ == "__main__":
    # generate dataset loader
    transform = transforms.Compose([transforms.ToTensor()])

    # ===== comment out this part if you just want to see the test result ======
    kd = KittiDataset(image_path='./data/train/image_left', label_path='./data/train/gt_image_left', transform=transform)

    # training
    model = CNN_LSTM().float()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.L1Loss()
    trainloader = tu.data.DataLoader(kd, batch_size=1, shuffle=False, num_workers=2)
    train_road_classifier(model, optimizer, trainloader, criterion)
    # ==========================================================================

    # testing
    md = CNN_LSTM().float()
    cmd = CNN_LSTM().float()

    md.load_state_dict(torch.load(os.path.join('./data/train/results-dets', 'md.pth')))
    cmd.load_state_dict(torch.load(os.path.join('./data/train/results-dets', 'md.pth')))

    image = Image.open('./data/test/image_left/uu_000041.jpg').convert('RGB')
    img_s = image.size

    # crop image at center with window size 600 * 160
    c_img = crop_image(image)
    c_img = generate_coord_channels(c_img)

    # scale original image to 600 * 160
    image = ImageOps.fit(image, (600, 160), Image.ANTIALIAS)
    image = generate_coord_channels(image)

    # turn image into torch tensor for prediction
    image = transform(image).view(1, 5, 160, 600)
    c_img = transform(c_img).view(1, 5, 160, 600)

    # predict
    outputs = md(image.float())
    outputs = outputs.detach().numpy().reshape((1, 600))

    coutputs = cmd(c_img.float())
    coutputs = coutputs.detach().numpy().reshape((1, 600))

    sv = visualize_segmentation(outputs)
    cv = visualize_segmentation(coutputs)

    final = np.array(reassemble(sv, cv, img_s))[:, :, 0]
    overlay_mask(cv2.imread('./data/test/image_left/umm_000087.jpg'), final)
