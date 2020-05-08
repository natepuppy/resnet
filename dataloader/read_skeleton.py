import numpy as np
import os
import pdb;

def read_skeleton(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []
            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_line = f.readline()
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, joint_line.split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence


def select_primary(frames):

    breakpoint()


def read_xyz(file, num_joints=25): #(joints X frames X 3) where 3 is x,y,z coordinates
    seq_info = read_skeleton(file)
    # if seq_info is None : return None #Couldn't read skeleton data because of incorrect data ('NaN')
    data = np.zeros((num_joints, seq_info['numFrame'], 3))
    frames = []
    for f, frame in enumerate(seq_info['frameInfo']):
        skeletons = []
        for b, body in enumerate(frame['bodyInfo']):
            skeleton = np.zeros((num_joints,3)) #get skeleton data for each body in frame
            # if b > 0 : return None #more than one body in frame
            for j, v in enumerate(body['jointInfo']):
                if j < num_joints:
                    if b == 0 : data[j, f, :] = [v['x'], v['y'], v['z']] #joints X frames X coordinantes
                    skeleton[j,:] = [v['x'], v['y'], v['z']]
            skeletons.append(skeleton)

        satisfied = False
        bad = False
        num_skeletons = len(skeletons)
        if num_skeletons == 1:
            frames.append(skeletons[0])
            continue

        keep = [] #ideally this would be a single value, but just to make sure more than one skeleton doesn't meet criterion
        for index, skeleton in enumerate(skeletons):
            x = skeleton[:,0]
            y = skeleton[:,1]
            x_range = x.max() - x.min()
            y_range = y.max() - y.min()
            if x_range > .8 * y_range:
                continue
            else:
                keep.append(skeleton)

        if len(keep) != 1: #you should only keep one skeleton per frame
            return None
        else:
            frames.append(keep[0])

    new_data = np.array(frames).swapaxes(0,1)
    if new_data.shape == data.shape:
        return new_data
    else:
        return data
# copy only single person skeletons
# find nturgb+d_skeletons/ -type f | grep -Ev 'A050|A054|A058|A051|A055|A059|A052|A056|A060|A053|A057' | xargs -i cp {} single_person/
if __name__ == '__main__':
    data_path = '/data0/NTU-RGB-D/nturgb+d_skeletons'
    test_skeleton = 'S001C001P001R002A010.skeleton' # two people
    # test_skeleton = 'S001C001P001R001A001.skeleton' # one people

    data = read_xyz(os.path.join(data_path, test_skeleton))
    print(data)
