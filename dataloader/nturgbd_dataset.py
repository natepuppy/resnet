#Created by John McPhie
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision
import os, sys
from .read_skeleton import *
from scipy import stats
from tqdm import tqdm
import pdb;
import time;
import datetime
import re
import warnings
import math
import numpy as np

sys.path.insert(0, os.path.abspath('..'))
from utils.visualization import visualize
from utils.image_processing import *

cross_subject_ids = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
class_descriptions = {
    1:'drink water', 2:'eat meal', 3:'brush teeth', 4:'brush hair', 5:'drop',
    6:'pick up', 7:'throw', 8:'sit down', 9:'stand up', 10:'clapping', 11:'reading',
    12:'writing', 13:'tear up paper', 14:'put on jacket', 15:'take off jacket',
    16:'put on a shoe', 17:'take off a shoe', 18:'put on glasses', 19:'take off glasses',
    20:'put on a hat/cap', 21:'take off a hat/cap', 22:'cheer up', 23:'hand waving',
    24:'kicking something', 25:'reach into pocket', 26:'hopping', 27:'jump up',
    28:'phone call', 29:'play with phone/tablet', 30:'type on a keyboard',
    31:'point to something', 32:'taking a selfie', 33:'check time (from watch)',
    34:'rub two hands', 35:'nod head/bow', 36:'shake head', 37:'wipe face',
    38:'salute', 39:'put palms together', 40:'cross hands in front',
    41:'sneeze/cough', 42:'staggering', 43:'falling down', 44:'headache',
    45:'chest pain', 46:'back pain', 47:'neck pain', 48:'nausea/vomiting', 49:'fan self'}

regex = re.compile("S(?P<setup_id>(\d{3}))C(?P<camera_id>(\d{3}))P(?P<subject_id>(\d{3}))R(?P<rep_num>(\d{3}))A(?P<class>(\d{3})).skeleton")
PROJECT_DIRECTORY = os.getcwd() + '/'
DATASETS_PATH = PROJECT_DIRECTORY + 'datasets/'
DESKTOP = '/'.join(os.getcwd().split('/')[:3]) + '/Desktop/'


class NTU_RGB_D(torch.utils.data.Dataset):
    """A custom dataset class for the NTURGB+D dataset with preprocessing built
    in. This dataset can read in skeleton, pytorch (pt), or RGB files.

    args:
        data_path: The absolute path to the directory where the dataset is
            found

        filetype: A string representing the type of dataset files to be read
            in.
            *Must be one of the following ('skeleton' | 'pt' | 'image').

        preprocess: A boolean representing whether to preprocess the dataset
            By default all the preprocessing steps are set to the same value
            as the preprocess boolean (e.g. preprocess=True means do all
            preprocessing).

        protocol: A string representing the evaluation protocol to be used.
            *Currently only used for loading skeleton files.
            *Must be one of the following:
                ('cross_subject' |'cross-view'| None)

        train: A boolean representing whether to load training or testing
            data.
            *Currently only used for loading skeleton files.

        rotate: A boolean representing if the skeletons should be rotated to
            face the 'camera.'

        zero_center: A boolean representing if the data should be zero
            centered.

        normalize: A boolean representing if the skeletons should be
            normalized. Normalized means that each joint is a unit vector
            centered on its parent. This will prevent visualization from
            working correctly unless the reconstruct=True flag is set.

        exclude_joints: A list of indices representing which joints to
            exclude.
            *Note the joints aren't actually removed, but set to 0.

        smooth: A boolean representing if the data should be smoothed out.

        group_joints: A boolean representing if the joints should be grouped
            by their proximity to each other. This will prevent
            visualization from working correctly.

        transform: A pytroch transformation to be done to the dataset.
    """
    def __init__(self, data_path, filetype='image', preprocess=False,
        protocol=None, train=True, rotate=True, zero_center=True, normalize=True,
        exclude_joints=[3,15,19,21,22,23,24], smooth=True, transform=None,
        group_joints=True):
        self.labels = []
        self.data = []
        self.filenames = []
        self.protocol = protocol
        self.rotate = rotate
        self.zero_center = zero_center
        self.normalize = normalize
        self.train = train
        self.exclude_joints = exclude_joints
        self.smooth = smooth
        self.image_dataset = None
        self.transform = transform
        self.group_joints = group_joints

        if filetype == 'skeleton':
            self.load_skeleton_data(data_path, preprocess=preprocess)
        elif filetype == 'pt':
            self.load_from_file(data_path + 'data.pt',  data_path + 'labels.pt', data_path + 'ids.pt', preprocess)
        elif filetype == 'image':
            self.load_images(data_path)
        else:
            error = "Could not load files of type '{}'\nFiletype must be ('skeleton'|'pt'|'image')".format(filetype)
            raise ValueError(error)


    def load_images(self, data_path):
        """Loads a dataset from folder of RGB images.

        args:
            data_path: A string representing the path to the image directory
        """
        #dir = data_path + '/' + self.protocol
        #dir += '/train/' if self.train else '/test/'
        transforms = T.Compose([T.ToTensor()])
        self.image_dataset = torchvision.datasets.ImageFolder(data_path, transform=transforms)
        self.num_classes = len(os.listdir(data_path))

    def save_as_file(self, data_filename=DATASETS_PATH+'data.pt', labels_filename=DATASETS_PATH+'labels.pt',
        ids_filename=DATASETS_PATH+'ids.pt'):
        """Saves the current dataset as a set of pytorch files for faster
        loading.

        args:
            data_filename: A string representing the filename/path to the
                pytorch file that will store the skeleton data.

            labels_filename: A string representing the filename/path to the
                pytorch file that will store the labels for each instance.

            ids_filename: A string representing the filename/path to the
                pytorch file that will store the ids of each skeleton data.
        """
        print("Saving...")
        torch.save([image for image in self.data], data_filename)
        torch.save([label for label in self.labels], labels_filename)
        torch.save([filename for filename in self.filenames], ids_filename)
        print("Done")

    def load_from_file(self, data_path, label_path, ids_path, preprocess=False):
        """Loads a previously saved dataset from pytorch files.

        args:
            data_filename: A string representing the filename/path to the
                pytorch file that contains the skeleton data.

            labels_filename: A string representing the filename/path to the
                pytorch file that contains the labels for each instance.

            ids_filename: A string representing the filename/path to the
                pytorch file that contains the ids corresponding to each
                skeleton data.

            preprocess: A boolean representing whether the data should be
                preprocessed.
        """
        print('Loading from file')
        self.data = torch.load(data_path)
        self.labels = torch.load(label_path)
        self.filenames = torch.load(ids_path)
        print('Successfully loaded {} files.'.format(len(self.data)))
        print()
        if preprocess:
            self._preprocess()
        else:
            self._resize()
        self.num_classes = len(np.unique(self.labels))

    def load_skeleton_data(self, data_path, preprocess=True):
        """Loads in a dataset from skeleton files.

        args:
            data_path: A string representing the path to the skeleton files.

            preprocess: A boolean representing whether the data should be
                preprocessed.
        """
        self.parse(data_path)
        if preprocess:
            self._preprocess()
        else:
            self._resize()
        self.num_classes = len(np.unique(self.labels))

    def add_data(self, file_info, data, label, filename):
        """Helper function for parse() used for sorting skeleton data based on
        protocol and train variables passed into dataset class. It only adds
        the data instance if it matches the training and cross subject/view
        protocols.

        args:
            file_info: A parsed regex expression holding the setup number,
                camera ID, performer ID, replication number, and action
                class label.

            data: Skeleton data (joints X frames X xyz).

            label: Label corresponding to the skeleton data instance.

            filename: Filename of the corresponding skeleton data instance.
        """
        include = False
        if self.protocol == 'cross_subject':
            subject_id = int(file_info.group('subject_id'))
            include = (self.train and subject_id in cross_subject_ids) or \
                (not self.train and subject_id not in cross_subject_ids)
        elif self.protocol == 'cross_view':
            camera_id = int(file_info.group('camera_id'))
            include = (self.train and camera_id in [2,3]) or \
                (not self.train and camera_id == 1)
        else:
            include = True

        if include:
            self.data.append(data)
            self.labels.append(label - 1)
            self.filenames.append(filename)

    def parse(self, data_path):
        """Parses a directory of skeleton files into pytorch tensors.

        args:
            data_path: A string representing the path to the skeleton files.
        """
        print('Reading Skeleton Data')
        ignored_samples = get_ignored(PROJECT_DIRECTORY)
        num = 0
        missing_data = []
        non_skeleton = []
        multi_person = []
        mispredicted = []
        for filename in tqdm(os.listdir(data_path), position=0):
            file_info = regex.fullmatch(filename)

            #comment these continues if you want the numbers to match paper
            if filename.split('.')[0] in ignored_samples: #files have missing data
                missing_data.append(filename)
                continue
            elif file_info.group('class') in ['0' + str(num + 50) for num in range(11)]: #two person class (50-60)
                multi_person.append(filename)
                continue
            elif not file_info: #Non skeleton file (e.g. .swp)
                non_skeleton.append(filename)
                continue

            file_path = data_path + '/' + filename
            data = read_xyz(file_path)
            label = int(file_info.group("class"))

            if data is not None:
                self.add_data(file_info, data, label, filename)
            else: #single person class that has two bodies
                # self.add_data(file_info, data, label, filename) #uncomment this if you want the numbers to match the paper
                mispredicted.append(filename)
                num += 1
        print("Num of non-skeleton files: ", len(non_skeleton))
        print("Num with missing data: ", len(missing_data))
        print("Num of multi-person class: ", len(multi_person))
        print("Num with mispredicted bodies:", len(mispredicted))
        print("Successfully read {} files.".format(len(self.data)))
        print()

    def _resize(self):
        """Resizes all the instances to have the same dimensions. This is done
            by resizing the frame dimesion to be the average frames in the
            dataset.
        """
        avg_frames = 87 #this is the average image frame length in the entire dataset
        for i in range(len(self.data)):
            image = self.data[i]
            self.data[i] = resize(image, width=avg_frames, height=len(image))

    def _preprocess(self):
        """Preprocesses data based on paramters passed into __init__()
        """
        print("Preprocessing...")
        invalid = []
        processed = []
        new_labels = []
        new_ids = []

        avg_frames = 87 #this is the average image frame length in the entire dataset
        for image, label, filename in tqdm(zip(self.data, self.labels, self.filenames), total=len(self.data),position=0):
            original = image.copy()
            image = resize(image, width=avg_frames, height=len(image))
            image = np.array(rotate(image)) if self.rotate else image
            image = zero_center(image) if self.zero_center else image
            image = normalize(image) if self.normalize else image
            image = smooth(image, (3,1)) if self.smooth else image
            image = exclude_joints(image, self.exclude_joints) if self.exclude_joints else image
            preprocessed = image.copy()
            image = group_joints(image) if self.group_joints else image

            processed.append(image)
            new_labels.append(label)
            new_ids.append(filename)

        self.data = processed
        self.labels = new_labels
        self.filenames = new_ids

    def save_images(self, dest):
        """Saves the current dataset as RGB images

        args:
            dest: the folder the dataset images should be saved to.
        """
        print('Writing images')
        for image_data, label, filename in tqdm(zip(self.data, [str(item) for item in self.labels], self.filenames), total=len(self.data), position=0):
            image = convert_to_rgb(image_data) #after normalization values are between -1 and 1, convert to between 0 and 255
            if not os.path.exists(dest + label):
                os.makedirs(dest + label)
            cv2.imwrite(dest + label + '/' + filename.strip('.skeleton') + '.png', image)

    def find_image(self, id):
        """Finds the data associated with a particular id.

        args:
            id: an id in the following format SsssCcccPpppRrrrAaaa. Where
                sss is the setup number, ccc is the camera ID, ppp is the
                performer (subject) ID, rrr is the replication number
                (1 or 2), and aaa is the action class label.
                Example: S001C001P001R001A001
        Return:
            A numpy image (joints X frames X xyz) if the skeleton file is
                found.
            None if the skeleton file is not found.
        """
        for data, filename in zip(self.data, self.filenames):
            if id == filename.split('.')[0] : return data
        return None

    def get_class_description(self, class_num):
        """Returns a human understandable description of a particular class.

        args:
            class_num: The numeric representation of the class (one indexed)
        """
        return class_descriptions[class_num]

    def get_label_description(self, label):
        """Returns a human understandable description of a particular label.

        args:
            label: zero indexed label for a data instance
        """
        return self.get_class_description(label + 1)

    def __len__(self):
        return len(self.labels) or len(self.image_dataset)

    def __iter__(self):
        return self

    def __getitem__(self, index): #(joints x frames x XYZ) -> (rgb x height x width)
        if self.image_dataset : return self.image_dataset.__getitem__(index)

        image = self.data[index] / 2 #change range to -.5 to .5
        image[np.isnan(image)] = 0
        x = torch.Tensor(image).permute(2,0,1)
        y = self.labels[index]
        if self.transform : x = self.transform(x)
        return x, y

def get_ignored(dir):
    """Gets a list of samples to ignore from the file named:
    'ignored_samples.txt'

    args:
        dir: Path to the folder containing 'ignored_samples.txt'
    """
    file = open(dir + 'ignored_samples.txt') #All files listed in the github, and a few more that we found
    return file.read().split()

#------------------------------------------Helper Functions------------------------------------------------------------
def gen_cross_subject(src=DATASETS_PATH+'nturgb+d_skeletons/', dest=DATASETS_PATH):
    """Generates the cross subject train/test data as RGB images.

    args:
        src: A string representing the path to the directory containing the
            skeleton files.

        dest: A string representing the path to a directory to store the RGB
            images (sorted by class) in.
    """
    print('Cross Subject')
    print('Loading raw skeleton data (train)...')
    raw_cross_subject_train = NTU_RGB_D(src, filetype='skeleton', train=True, protocol='cross_subject')
    raw_cross_subject_train.save_images(dest + '/raw/cross_subject/train/')

    print('Loading raw skeleton data (test)...')
    raw_cross_subject_test = NTU_RGB_D(src, filetype='skeleton', train=False, protocol='cross_subject')
    raw_cross_subject_test.save_images(dest + '/raw/cross_subject/test/')

    print('Loading preprocessed skeleton data (train)...')
    pre_cross_subject_train = NTU_RGB_D(src, filetype='skeleton', preprocess=True, train=True, protocol='cross_subject')
    pre_cross_subject_train.save_images(dest + '/preprocessed/cross_subject/train/')

    print('Loading preprocessed skeleton data (test)...')
    pre_cross_subject_test = NTU_RGB_D(src, filetype='skeleton', preprocess=True, train=False, protocol='cross_subject')
    pre_cross_subject_test.save_images(dest + '/preprocessed/cross_subject/test/')

def gen_cross_subject_pt(src=DATASETS_PATH+'nturgb+d_skeletons/', dest=DATASETS_PATH):
    """Generates the cross subject train/test data as pytorch files.

    args:
        src: A string representing the path to the directory containing the
            skeleton files.

        dest: A string representing a path to the directory to store the
            pytorch files in.
    """
    train_dir = dest + '/raw/cross_subject/train/'
    test_dir = dest + '/raw/cross_subject/test/'
    if not os.path.exists(train_dir) : os.makedirs(train_dir)
    if not os.path.exists(test_dir) : os.makedirs(test_dir)

    cross_subject_train = NTU_RGB_D(src, filetype='skeleton', preprocess=False, train=True, protocol='cross_subject')
    cross_subject_train.save_as_file(train_dir + 'data.pt', train_dir + 'labels.pt', train_dir + 'ids.pt')

    cross_subject_test = NTU_RGB_D(src, filetype='skeleton', preprocess=False, train=False, protocol='cross_subject')
    cross_subject_test.save_as_file(test_dir + 'data.pt', test_dir + 'labels.pt', test_dir + 'ids.pt')

    pre_path = dest + '/preprocessed/cross_subject/'
    train_dataset = NTU_RGB_D(train_dir, filetype='pt', preprocess=True)
    train_dataset.save_as_file(pre_path + 'train/data.pt', pre_path + 'train/labels.pt', pre_path + 'train/ids.pt')

    test_dataset = NTU_RGB_D(test_dir, filetype='pt', preprocess=True)
    test_dataset.save_as_file(pre_path + 'test/data.pt', pre_path + 'test/labels.pt', pre_path + 'test/ids.pt')

def write_images():
    """Simple helper function for writing raw dataset (pytorch files) as RGB
    images.
    """
    dataset = NTU_RGB_D(DATASETS_PATH, filetype='pt', preprocess=False)
    dataset.save_images(DATASETS_PATH + 'raw/all/')

def write_image(image, filename='image.png'):
    """Writes a given data instance as an RGB file to the desktop.

    args:
        filename: A string representing the name of the file to be saved to
            the desktop.
    """
    if type(image) != np.ndarray : image = np.array(image)
    if image.shape[-1] != 3 and image.shape[-1] != 1:
        print('Please make sure dimensions are: (height,width,channels)')
    else:
        cv2.imwrite(DESKTOP + filename, image)
        print('wrote file')

def see_preprocess(image, label):
    """Shows the effects of some preprocessing steps.

    args:
        image: A skeleton data numpy array (JOINTS x FRAMES x XYZ)
        label: An integer representing the zero indexed class label.
    """
    visualize(image, media='video', filename=DESKTOP+'original.mp4', zoom=.5)
    image = np.array(rotate(image))
    image = normalize(image)
    image = exclude_joints(image, [3,15,19,21,22,23,24])
    visualize(reconstruct(image), zoom=3.5, media='video', filename=DESKTOP+'preprocessed.mp4')
    description = class_descriptions[label + 1]
    print('Class:', description)

def test():
    """A function used to demonstrate a simple use case of the custom dataset.
    """
    dataset = NTU_RGB_D(DATASETS_PATH + 'raw/cross_subject/train/', filetype='pt', preprocess=False)
    #dataset = NTU_RGB_D(DATASETS_PATH + 'nturgb+d_skeletons/', filetype='skeleton', preprocess=False)
    #dataset = NTU_RGB_D(DATASETS_PATH + 'preprocessed/cross_subject/train/', filetype='image', preprocess=False)

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,pin_memory=False)

    for batch, labels in loader:
        for image, label in zip(batch, labels):
            description = dataset.get_label_description(label.item())
            print(description)
            visualize(image, zoom=.5)
            #visualize(image, zoom=.5, media='video')
            exit()
