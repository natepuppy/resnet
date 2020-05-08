#Created by John McPhie
import train
import utils.image_processing as IP
import utils.visualization as V
import dataloader.nturgbd_dataset as D
import re, os, torch

regex = re.compile("S(?P<setup_id>(\d{3}))C(?P<camera_id>(\d{3}))P(?P<subject_id>(\d{3}))R(?P<rep_num>(\d{3}))A(?P<class>(\d{3})).skeleton")
PROJECT_DIRECTORY = os.getcwd() + '/'
DATASETS_PATH = PROJECT_DIRECTORY + 'datasets/'
DESKTOP = '/'.join(os.getcwd().split('/')[:3]) + '/Desktop/'
TEST_ID = 'S001C001P001R001A001'

dataset = None

def get_instance(as_numpy=True, id=None, random=True):
    """Gets a data instance and label from the raw nturgb+d dataset.

        args:
            as_numpy: A boolean indicating whether the instance should be
                converted to a numpy array with dims (JOINTSxFRAMESxXYZ) as
                opposed to a pytorch tensore with dims (XYZxJOINTSxFRAMES).
                It also returns the label as a description instead of a number.

            id: A string representing the id of the skeleton data file to get.

            random: A boolean representing if the data instance should be
                random as opposed to deterministic (ordered by id).
    """
    global dataset
    if not dataset:
        dataset = D.NTU_RGB_D(DATASETS_PATH+'raw/all/', filetype='pt', preprocess=False)
    if id : return dataset.find_image(id)

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=random,
        num_workers=1,pin_memory=False)

    for batch, labels in loader:
        for image, label in zip(batch, labels):
            if as_numpy:
                image = image.permute(1,2,0).numpy()
                label = dataset.get_label_description(label.item())
            return image, label
