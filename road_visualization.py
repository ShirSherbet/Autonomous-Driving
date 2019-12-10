import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import lstsq
import skimage.io
import open3d as o3d
from mpl_toolkits import mplot3d


def compute_3d_location(depth, px, py, f):
    """
    Project 2: Qestion 4 & 5
    Return a matrix of size (h x w x 3) such that m[y, x] = [X, Y, Z]

    :param depth: depth map of the stereo image pair
    :type depth: numpy array
    :param px: principal point x of camera
    :type px: float
    :param py: principal point y of camera
    :type py: float
    :param f: focal length
    :type f: float
    :return: matrix of the XYZ coordinates in world coordinate system for each pixel
    :rtype: numpy array
    """

    xy_coord = make_coordinates_matrix(depth.shape)
    z_coord = depth.reshape((depth.shape[0], depth.shape[1], 1))

    # calculate X Y coordinates: Z * (x - px) / f
    #                            Z * (y - py) / f
    xy_coord = (xy_coord - [px, py]) / f
    xy_coord = np.multiply(xy_coord, np.concatenate((z_coord, z_coord), axis=2))

    location = np.concatenate((xy_coord, z_coord), axis=2)
    return location


def make_coordinates_matrix(im_shape):
    """
    Helper for compute_3D_location
    Return a matrix of size (im_shape[0] x im_shape[1] x 2) such that m[y, x] = [y, x]
    :param im_shape: (y, x, channel)
    :type im_shape: tuple
    :return: xy coordinate matrix of the image
    :rtype: numpy array
    """
    range_x = np.arange(0, im_shape[1], 1)
    range_y = np.arange(0, im_shape[0], 1)
    axis_x = np.repeat(range_x[np.newaxis, ...], len(range_y), axis=0)
    axis_y = np.repeat(range_y[..., np.newaxis], len(range_x), axis=1)

    return np.dstack((axis_y, axis_x))

def fit_plane(road_loc):
    """
    Project 2: Qestion 4
    Fit a plane based on the 3D road point cloud
    :param road_loc: a matrix of size (num_road_pixel x 3)
    :type road_loc: numpy array
    :return:
    :rtype:
    """
    # use LSE (least square estimation) to find the best fitting
    # num_road_pixel x 3 where A[i] = [Xi, Yi, 1]
    A = np.c_[road_loc[:, 0], road_loc[:, 1], np.ones(road_loc.shape[0])]
    # run lse to get coefficients
    C, _, _, _ = lstsq(A, road_loc[:, 2])
    plane = (C[0], C[1], -1, C[2])
    nn = np.linalg.norm(plane)
    plane = plane / nn

    return plane

def compute_distance(points, plane):
    """
    Helper for RANSAC
    return a 1d array of size N (number of points) containing the distance of a point to the given plane
    """
    # convert 3D point to Homogenous coordinate
    points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    # calculate distance  = |Ax + By + Cz + d| / sqrt(A^2 + B^2 + C^2)
    dists = np.abs(np.dot(points, plane)) / np.sqrt(np.sum(plane[:3] ** 2))
    return dists

def RANSAC(locations, iters=3000, thresh=0.02):
    """
    Project 2: Qestion 4
    RANSAC to estimate a plane roboust to outliers
    :param locations: N x 3 3D points
    :type locations: numpy array
    :param iters: number of iterations for RANSAC
    :type iters: int
    :param inlier_thresh: point-to-plane distance threshold
    :type inlier_thresh: float
    :return: 4D plane vector and a list of inlier indices in the points
    :rtype:
    """
    max_inlier_num = -1
    best_inlier_idxs = None

    # number of points
    N = locations.shape[0]

    for i in range(iters):
        # randomly generate 5 points to fit plane
        random_idx = np.random.choice(N, 5, replace=False)
        random_points = locations[random_idx, :]

        # fit plane
        curr_plane = fit_plane(random_points)

        # calculate distance and find inliers
        dists = compute_distance(locations, curr_plane)
        curr_inlier_idxs = np.where(dists < thresh)[0]
        inlier_num = curr_inlier_idxs.shape[0]

        # update best match
        if inlier_num > max_inlier_num:
            max_inlier_num = inlier_num
            best_inlier_idxs = curr_inlier_idxs

    # using the bset match we find so far to fit the final plane
    best_points = locations[best_inlier_idxs, :]
    final_plane = fit_plane(best_points)

    # final filter for inliers
    dists = compute_distance(locations, final_plane)
    inlier_list = np.where(dists < thresh)[0]

    # [a,b,c,d] -> [a,b,-1,d]
    final_plane = final_plane / -final_plane[2]
    return final_plane, inlier_list


def filter_pixel(location, road_mask):
    """
    Helper function
    Filter pixel locations and return only the road pixels
    """
    # pre-processing mask for pixel classification
    road_mask[np.where(road_mask < 0.5)] = 0
    road_mask[np.where(road_mask >= 0.5)] = 1
    road_mask = road_mask.astype(int)
    # filter location points and save only road pixels
    # num_road_pixel x 3
    road_loc = np.copy(location[np.where(road_mask == 1)])

    return road_loc

def visulaize_plane(X, Y, Z, road_loc):
    """
    Visualize the estimated plane with a road point cloud
    """
    road_fig = plt.figure()
    road_ax = plt.axes(projection='3d')
    road_ax.plot_surface(Z, Y, X, rstride=1, cstride=1, alpha=0.2)
    road_ax.scatter(road_loc[:,2], road_loc[:,1], road_loc[:,0], cmap='viridis', marker='.', c=road_loc[:,2])

    road_ax.set_xlabel('z')
    road_ax.set_ylabel('y')
    road_ax.set_zlabel('x')
    plt.title('Road 3D Points with Plane', fontsize=16)
    plt.show()

def visualize_all_with_plane(X, Y, Z, location, image):
    """
    Project 2: Qestion 5
    Visualize all the pixel 3D locations and the estimated plane
    """
    XX = X.flatten().reshape((-1,1))
    YY = Y.flatten().reshape((-1,1))
    ZZ = Z.flatten().reshape((-1,1))
    road_plane_coord = np.concatenate((XX, YY, ZZ), axis=1)

    # create point cloud for road plane
    road_pcd = o3d.geometry.PointCloud()
    road_pcd.points = o3d.utility.Vector3dVector(road_plane_coord)
    road_pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0] for i in range(XX.shape[0])])

    # create point cloud for all image pixels
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(location.reshape((-1,3)))
    pcd.colors = o3d.utility.Vector3dVector(image.reshape((-1,3))/255)

    o3d.visualization.draw_geometries([road_pcd, pcd])

def compute_XYZ(points, plane):
    """
    Helper for visualization
    """
    # find boundary
    maxx = np.max(points[:, 0])
    minx = np.min(points[:, 0])
    maxy = np.max(points[:, 1])
    miny = np.min(points[:, 1])
    # create XY meshgrid
    X, Y = np.meshgrid(np.arange(minx - 50, maxx + 100, 5), np.arange(miny - 50, maxy + 100, 5))
    # evaluate Z on XY meshgrid
    Z = plane[0] * X + plane[1] * Y + plane[3]
    return X,Y,Z

if __name__ == "__main__":
    F = 7.215377000000e+02
    PX, PY = 6.095593000000e+02, 1.728540000000e+02

    depth = skimage.io.imread('./Output Images/depth_umm_000087.png', as_gray=True)
    # Q4: Fit plane
    loc = compute_3d_location(depth, PX, PY, F)
    mask = skimage.io.imread('./Output Images/road_segmentation_mask_umm_000087.png', as_gray=True) / 255
    road_loc = filter_pixel(loc, mask)
    plane, inl = RANSAC(road_loc)

    # Q5: Visualize plane
    left_img = skimage.io.imread('./data/test/image_left/umm_000087.jpg')
    inl = road_loc[inl]
    X, Y, Z = compute_XYZ(inl, plane)

    visualize_all_with_plane(X, Y, Z, loc, left_img)
    visulaize_plane(X, Y, Z, road_loc)
