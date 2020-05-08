#Created by John McPhie
import torch
import math
import cv2
import numpy as np

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d

parent_joints = [1, 1, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18, 1, 7, 7, 11, 11]
parent_first = [1, 0, 20, 16, 12, 17, 13, 18, 19, 14,15, 2, 3, 4, 5, 6, 7, 21, 22, 8, 9, 10, 11, 24, 23]
root_joint = 1 #joint 2 in the original pdf (https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shahroudy_NTU_RGBD_A_CVPR_2016_paper.pdf)
lines = [(joint, parent_joints[joint]) for joint in range(25)] #used for plotting (ex. plot(joints, lines))

#Note: for the purpose, and image is a numpy array with dimensions (NUM_JOINTS, NUM_FRAMES, XYZ)

def radians_to_x_axis(points):
    """Finds the radians to the X-axis.

    args:
        points: A numpy array with dimensions (n, 3)

    returns:
        A numpy array with the n values.
    """
    radians = np.arctan(np.dot(points, [0,0,1])/np.dot(points, [1,0,0]))
    return radians + np.pi #Add pi(180 degrees) so they face forward instead of backwards

def plot(points, lines=None):
    """Plots the given points on a 3D plane.

        args:
            points: A numpy array with dimensions (n, m, 3)

            lines: A list of tuples containing the indices (with respect to the
                points paramter) indicating which points should be connected
                with a line.
    """
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.view_init(azim=-90, elev=90)
    ax.scatter(x, y, z, 'ro-')

    if lines:
        for endpoints in lines:
            xs = x[endpoints[0]], x[endpoints[1]]
            ys = y[endpoints[0]], y[endpoints[1]]
            zs = z[endpoints[0]], z[endpoints[1]]
            line = plt3d.art3d.Line3D(xs, ys, zs)
            ax.add_line(line)
    plt.show()

def rotation_matrix(degrees, axis, unit='degrees'):
    """Returns a 3D rotation matrix.

        args:
            degrees: The degrees to rotate

            axis: A string representing the axis to rotate around
                ('x' | 'y' | 'z')
    """
    theta = np.radians(degrees) if unit == 'degrees' else degrees

    if axis == 'x':
        r = np.array([
            [1,0,0],
            [0,np.cos(theta),-np.sin(theta)],
            [0,np.sin(theta),np.cos(theta)],
        ])
    elif axis == 'y':
        r = np.array([
            [np.cos(theta),0,np.sin(theta)],
            [0,1,0],
            [-np.sin(theta),0,np.cos(theta)],
        ])
    elif axis == 'z':
        r = np.array([
            [np.cos(theta),-np.sin(theta),0],
            [np.sin(theta),np.cos(theta),0],
            [0,0,1],
        ])
    return r

def rotate(image):
    """Rotates a skeleton instance so that it faces forward.

        args: A numpy array with dimensions: (JOINTS, FRAMES, XYZ)
    """
    image = torch.Tensor(image).clone()
    parent = image[parent_joints].clone()
    zero_centered = image - parent

    right_shoulder = zero_centered[4]
    left_shoulder = zero_centered[8]
    shoulders = right_shoulder - left_shoulder

    thetas = radians_to_x_axis(shoulders)
    for i, theta in enumerate(thetas): #rotate each frame
        skeleton = image[:,i]
        r = rotation_matrix(theta, 'y', 'radians') #around y-axis
        image[:,i] = np.matmul(r, skeleton.T).T

    return image

def zero_center(image):
    """Centers a skeleton instance on the origin (0,0,0). Assumes root joint at
        index 1.

        args:
            image: A numpy array with dimensions: (JOINTS, FRAMES, XYZ)
    """
    origin = image[root_joint]
    return image - origin

def joint_center(image):
    """Centers each joint on their parent joint.

        args:
            image: A numpy array with dimensions: (JOINTS, FRAMES, XYZ)
    """
    parent = image[parent_joints].copy()
    return image - parent

def normalize(image):
    """Parent centers and normalizes a skeleton instance so that all bones are
    unit vectors coming out of the origin.

        args:
            image: A numpy array with dimensions: (JOINTS, FRAMES, XYZ)
    """
    num_frames = len(image[0])

    joint_centered = joint_center(image)
    distances = np.linalg.norm(joint_centered, axis=2)
    distances[root_joint] = np.ones(num_frames) #root joint has itself as its parent (don't divide by zero)

    if distances.min() == 0 : return None #invalid image
    norm = np.divide(joint_centered, distances[:, :, np.newaxis]) #newaxis for broadcasting: (25, frames, (xyz)) / (25, frames, 1)

    return norm

def reconstruct(image):
    """Reconstructs skeleton from normalized pieces. After being zero centered
        on parent joints, all joints come out of the origin and are hard to
        visualize. This function reconstructs the skeleton so that
        visualization is possible.

        args:
            image: A numpy array with dimensions: (JOINTS, FRAMES, XYZ)
    """
    reconstructed = image.copy()
    for joint in parent_first: #You have to make them in order from parent to child
        p_joint = parent_joints[joint]
        reconstructed[joint] = reconstructed[joint] + reconstructed[p_joint]
    return reconstructed

def convert_to_rgb(image):
    """Changes values to be integers between 0 and 255.

        args:
            image: A numpy array with dimensions: (JOINTS, FRAMES, XYZ)
    """
    if type(image) != np.ndarray : image = np.array(image)
    old_range = (image.max() - image.min())
    new_range = (255 - 0)
    return ((((image - image.min()) * new_range) / old_range) + 0).astype(np.uint8)

def bilateralFilter(image, k_size, sigma_color, sigma_space):
    """Performs a bilateral filter (blurring) on the given image.

        args:
            image: A numpy array with dimensions: (m, n, RGB) or (J, F, XYZ)

            k_size:

            sigma_color:

            sigma_space:
    """
    copy = np.zeros_like(image)
    for j,joint in enumerate(image):
        joint_image = np.array([joint] * 25)
        result = cv2.bilateralFilter(joint_image, k_size, sigma_color, sigma_space)
        copy[j] = result.mean(axis=0)
    return copy

def smooth(image, kernel=(3,1)):
    """Smoothes a given image.

        args:
            image: A numpy array with dimensions: (m, n, RGB) or (J, F, XYZ)
            kernel: The convolutional kernel size to be used for smoothing.
    """
    # return cv2.blur(image, kernel)
    # return cv2.GaussianBlur(image, kernel,0)
    # from .visualization import visualize
    result = bilateralFilter(image, 6, 50, 200)
    return result


def exclude_joints(image, joints):
    """Excludes the given joints from a skeleton instance (JOINTS x FRAMES x
        XYZ) by setting them to NaN.

        args:
            image: A numpy array with dimensions: (m, n, RGB) or (J, F, XYZ)
            joints: A list of indices (rows) to be removed.
    """
    adjusted = image.copy()
    adjusted[joints] = np.nan
    return adjusted

def group_joints(image):
    """Rearranges joints so they are grouped with other joints close in
        proximity.

        args:
            image: A numpy array with dimensions: (m, n, RGB) or (J, F, XYZ)
    """
    result = np.zeros_like(image)
    left_arm = [20,4,5,6,7]
    right_arm = [20,8,9,10,11]
    left_leg = [1,0,12,13,14]
    right_leg = [1,0,16,17,18]
    torso = [2,8,20,4,1]
    result[0:5] = image[torso]
    result[5:10] = image[left_arm]
    result[10:15] = image[right_arm]
    result[15:20] = image[left_leg]
    result[20:25] = image[right_leg]
    return result

def resize(image, width, height):
    """Resizes an image to have the given width and height.

        args:
            image: A numpy array with dimensions: (m, n, RGB) or (J, F, XYZ)
            width: An integer representing the new width of the image (pixels)
            height: An integer representing the new height of the image (pixels)
    """
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
