#Created by John McPhie
import cv2
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
import mpl_toolkits.mplot3d as plt3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import Divider, Size
from mpl_toolkits.axes_grid1.mpl_axes import Axes
import numpy as np
from tqdm import tqdm
import os
try:
    from .image_processing import parent_joints, convert_to_rgb, reconstruct, rotation_matrix
    from .read_skeleton import read_skeleton
except ImportError as error:
    from read_skeleton import read_skeleton
    from image_processing import parent_joints, convert_to_rgb, reconstruct, rotation_matrix

RESULTS_PATH = '/'.join(os.getcwd().split('/')[:3]) + '/Desktop/'
PROJECT_DIRECTORY = os.getcwd() + '/'
DATASETS_PATH = PROJECT_DIRECTORY + 'datasets/'

def rgb_image(image,filename=None):
    """Converts skeleton data (JOINTS x FRAMES x XYZ) into an RGB image. If no
        filename is passed in, the RGB image is simply displayed. If a filename
        is provided, the image is saved with that filename.

        args:
            image: Numpy array with dimensions (NUM_JOINTS x NUM_FRAMES x XYZ)

            filename (optional): A string representing a filename for saving
            the RGB image.
    """
    rgb = convert_to_rgb(image)
    if filename:
        cv2.imwrite(filename, rgb)
    else:
        plt.imshow(rgb[...,::-1]) #RGB -> BGR
        plt.show()

def align_axes(image):
    """Aligns the skeleton axes so that the skeleton is seen from the correct
        perspective (vertically insted of horizontally). In matplotlib the
        z-axis is the up direction, but in xyz skeleton coordinates the y-axis
        is the up direction. 
        *Note: This does an inplace operation.

        args:
            image: A skeleton data instance (JOINTS x FRAMES x XYZ)
    """
    center = image[1]
    image -= center
    num_joints, num_frames = image.shape[:-1]
    r = rotation_matrix(90, 'x')
    for j in range(num_joints):
        for f in range(num_frames):
            point = image[j,f]
            image[j,f] = np.dot(r,point)
    image += center

def visualize(image, zoom=0, media=None, filename=None, reconstruct_skeleton=False):
    """Visualizes a set of skeletal joints in 3d space.

    args:
        image: A skeleton data instance (JOINTS x FRAMES x XYZ)

        zoom: An integer representing a zoom factor
            *Note: 0 means none, + means closer, - means furthers

        media: A string (None | 'image' | 'video' ) indicating how the
            skeleton is to be represented.

            None: The first frame will be displayed on screen with
                matplotlib.
            'image': The first frame will be saved as a .png file on the
                desktop unless the filename is specified.
            'video': All the frames will be saved as an .mp4 file on the
                desktop unless the filename is specified.

        filename: The name of the file for a image or video file.

        reconstruct: Boolean used to indicated that the skeleton needs to be
            reconstructed from parent-centered joints.
    """
    zoom = max(-zoom+1, 1) if zoom <= 0 else 1/(zoom+1)
    image = np.array(image)
    if reconstruct_skeleton : image = reconstruct(image)
    align_axes(image)
    original_shape = image.shape
    if original_shape[0] == 3 : image = image.transpose(1,2,0)
    try:
        parent = image[parent_joints].copy()
        joints = np.array(image).copy()
        p_joints = np.array(parent).copy()
        excluded_joints = np.unique(np.where(np.isnan(joints))[0])
        num_frames = len(image[0])
        xview = joints[1,0,0] #1 for center mass, 0 for first frame, and 0 for x coordinate
        yview = joints[1,0,1]
        zview = joints[1,0,2]

        fig = plt.figure()
        ax = Axes3D(fig)

        def init():
            ax.view_init(azim=90, elev=0)
            return fig,

        def animate(frame):
            ax.clear()
            ax.set_xlim([xview - zoom, xview + zoom])
            ax.set_ylim([yview - zoom, yview + zoom])
            ax.set_zlim([zview - zoom, zview + zoom])
            ax.autoscale(enable=False)
            points = np.delete(joints, excluded_joints, axis=0)
            ax.scatter(points[:,frame,0], points[:,frame,1], points[:,frame,2], 'ro-')

            for i in range(len(joints)):
                if i in excluded_joints : continue
                xs = joints[i,frame, 0], p_joints[i,frame, 0]
                ys = joints[i,frame, 1], p_joints[i,frame, 1]
                zs = joints[i,frame, 2], p_joints[i,frame, 2]
                line = plt3d.art3d.Line3D(xs, ys, zs)
                ax.add_line(line)
            return fig,

        if not media:
            init()
            animate(0)
            plt.show()
        else:
            if media == 'video':
                filename = filename or RESULTS_PATH + 'video.mp4'
                anim = animation.FuncAnimation(fig, animate, init_func=init, frames=num_frames, interval=20, blit=True)
                anim.save(filename, fps=30, extra_args=['-vcodec', 'libx264'])
            else:
                filename = filename or RESULTS_PATH + 'image.png'
                init()
                animate(0)
                plt.savefig(filename)
    except Exception as error:
        print(error)
        print('Please make sure your dimensions are correct: (joints, frames, xyz) or (25, num_frames, 3)')
        print('Current dimensions:', original_shape)

def superimpose(id, skeleton_dir=DATASETS_PATH+'nturgb+d_skeletons/',
    video_dir=PROJECT_DIRECTORY+'videos/rgb/', dest=RESULTS_PATH+'rgb_skeleton.avi'):
    """Superimposes the skeletal information from a skeleton file on the
        the corresponding RGB video.

        args:
            id: An string in the following format SsssCcccPpppRrrrAaaa. Where
                sss is the setup number, ccc is the camera ID, ppp is the
                performer (subject) ID, rrr is the replication number (1 or 2),
                and aaa is the action class label. Example: S001C001P001R001A001

            skeleton_dir: A string representing the path to a directory
                containing .skeleton files.

            video_dir: A string representing the path to a directory
                containing the RGB videos corresponding to the .skeleton files.

            dest: A string representing a directory to store the results in.
    """
    skeleton_path = skeleton_dir + id + '.skeleton'
    video_path = video_dir + id + '_rgb.avi'
    image_info = read_skeleton(skeleton_path)
    frames = []

    data = np.zeros((25, image_info['numFrame'], 2, 3))
    for f, frame in enumerate(image_info['frameInfo']):
        skeletons = {}
        for b, body in enumerate(frame['bodyInfo']):
            skeleton = np.zeros((25,3))
            lines = []
            circles = []
            for j, joint in enumerate(body['jointInfo']):
                k_index = get_parent_index(j)
                dx = int(joint['colorX'])
                dy = int(joint['colorY'])
                joint2 = body['jointInfo'][k_index]
                dx2 = int(joint2['colorX'])
                dy2 = int(joint2['colorY'])
                lines.append({'start': (dx,dy), 'end': (dx2,dy2)})
                circles.append({'center':(dx, dy)})
                skeleton[j,:] = np.array([joint['x'], joint['y'], joint['z']])
                data[j, f, b, :] = np.array([joint['x'], joint['y'], joint['z']])
            skeletons[b] = {'xyz':skeleton, 'color':{'lines': lines, 'circles':circles}}
        frames.append(skeletons)
    draw_skeleton(video_path, frames, dest)

def draw_skeleton(input_video, skeleton_info, out_file):
    """Draws the skeletons on the input video based off the given information.

    args:
        input_video: A string representing the path to an rgb video.
        skeleton_info: A list of the following for each frame in the video
        file: {'xyz':skeleton, 'color':{'lines': lines, 'circles':circles}}
    """
    major_ver, minor_ver, subminor_ver = (cv2.__version__).split('.')
    video = cv2.VideoCapture(input_video)
    if (not video.isOpened()):
        print("Error opening video stream or file:", input_video)
        return

    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    frame_rate = int(video.get(5))
    out = cv2.VideoWriter(out_file,cv2.VideoWriter_fourcc('M','J','P','G'), frame_rate, (frame_width,frame_height))
    green = (0, 255, 0)
    blue = (255, 0, 0)
    colors = [green, blue]
    thickness = 5

    for skeletons in skeleton_info: #frames is a list of maps
        ret, frame = video.read()
        for id, skeleton in skeletons.items():
            index = 0
            for line, circle in zip(skeleton['color']['lines'], skeleton['color']['circles']):
                frame = cv2.line(frame, line['start'], line['end'], colors[id], thickness)
                frame = cv2.circle(frame, circle['center'], 5, (0,0,255), -1)
                index += 1
        if not ret : break
        out.write(frame)

    video.release()
    cv2.destroyAllWindows()

def get_parent_index(child_index):
    """Returns the parent index of a given joint.

    args:
        child_index: The 0 based index of a particular joint.
    """
    return parents[child_index + 1] - 1

parents = {
    1: 1, #center
    2: 1,
    21: 2,
    3: 21,
    4: 3,
    5: 21, #Left shoulder ->
    6: 5,
    7: 6,
    8: 7,
    22: 8,
    23: 8, #Left thumb
    9: 21, #Right shoulder ->
    10: 9,
    11: 10,
    12: 11,
    24: 12,
    25: 12, #Right thumb
    13: 1, #Left hip
    14: 13,
    15: 14,
    16: 15, #Left foot
    17: 1, #Right hip
    18: 17,
    19: 18,
    20: 19 #Right foot
}
