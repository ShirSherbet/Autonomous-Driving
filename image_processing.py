import numpy as np
import cv2
from matplotlib import pyplot as plt
import skimage.io

def compute_disparity(left, right, p):
    """
    Project 2: Question 1
    Method 1

    :param left: the left image in the stereo image pairs
    :type left: numpy array
    :param right: the right image in the stereo image pair
    :type right: numpy array
    :param p: patch size
    :type p: int
    :return: disparity map of the image pair
    :rtype: numpy array of the same shape as the input images
    """
    # max disparity to check
    max_d = 16

    disp_map = np.zeros(left.shape)

    # sizes
    im_height, im_width = left.shape
    p_half = p // 2

    # zero-padding: N + p - 1 * M + p - 1 for left and right image
    padded_shape = im_height + p - 1, im_width + p - 1

    padded_im_left = np.zeros(padded_shape)
    padded_im_left[p_half:(im_height + p_half), p_half:(im_width + p_half)] = left

    padded_im_right = np.zeros(padded_shape)
    padded_im_right[p_half:(im_height + p_half), p_half:(im_width + p_half)] = right

    # matrix of vectorized patches for right image for fast retrieval
    r_patch_vectorized = np.zeros((im_height, im_width, p*p))

    for i in range(im_height):
        for j in range(im_width):
            l_patch = padded_im_left[i: i + p, j: j + p].reshape(-1)
            r_patch = padded_im_right[i: i + p, j: j + p].reshape(-1)

            r_patch_vectorized[i, j, :] = r_patch

            # calculate absolute intensity difference
            start = max(0, j - max_d)
            abs_diff = np.abs(np.copy(r_patch_vectorized[i, start: j + 1, :]) - l_patch)
            ssd = np.sum(abs_diff, axis=1)

            # find the best match
            ssd = np.flip(ssd, 0)
            min_d = np.argmin(ssd)

            disp_map[i, j] = min_d

    cv2.normalize(disp_map, disp_map, 0, 255, cv2.NORM_MINMAX)
    return disp_map

def compute_disparity_by_cv2(left, right, block_size):
    """
    Project 2: Question 1
    Method 2

    :param left: the left image in the stereo image pairs
    :type left: numpy array
    :param right: the right image in the stereo image pair
    :type right: numpy array
    :return: disparity map of the image pair
    :rtype: numpy array of the same shape as the input images
    """
    stereoMatcher = cv2.StereoBM_create()
    stereoMatcher.setMinDisparity(4)
    stereoMatcher.setNumDisparities(128)
    stereoMatcher.setBlockSize(block_size)
    stereoMatcher.setSpeckleRange(16)
    stereoMatcher.setSpeckleWindowSize(10)
    disp = stereoMatcher.compute(left, right)

    kernel = np.ones((5, 5), np.float32) / 25
    disp = cv2.filter2D(disp, -1, kernel)
    return disp


def compute_depth(disp_map, f, t):
    """
    Project 2: Qestion 2

    :param disp_map: disparity map of the stereo image pair
    :type disp_map: numpy array
    :param f: focal length
    :type f: float
    :param t: baseline
    :type t: float
    :return: depth map of the image pair
    :rtype: numpy array of the same shape as the input disp_map
    """
    depth = np.divide(-f * t, disp_map, where=disp_map != 0)
    cv2.normalize(depth, depth, 0, 255, cv2.NORM_MINMAX)
    return depth

if __name__ == "__main__":
    F = 7.215377000000e+02
    T = 0.54 * 1000
    PX, PY = 6.095593000000e+02, 1.728540000000e+02

    left_img = cv2.imread('./data/test/image_left/umm_000087.jpg', cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread('./data/test/image_right/umm_000087.jpg', cv2.IMREAD_GRAYSCALE)

    # Q1: Compute disparity
    # Method 1
    # disp = compute_disparity(left_img, right_img, 10)
    # Method 2
    disp = compute_disparity_by_cv2(left_img, right_img, 5)

    plt.imshow(disp, 'gray')
    plt.title('Disparity', fontsize=16)
    plt.show()

    # Q2: Compute depth
    depth = compute_depth(disp, F, T)
    plt.imshow(depth, 'gray')
    plt.title('Depth', fontsize=16)
    plt.show()
    skimage.io.imsave('./Output Images/depth_umm_000087.png', depth.astype('uint8'))
