import pathlib
from PIL import Image
import numpy as np
from utils.utils import *
from scipy.spatial.transform import Rotation

"""
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.

"""


def getDataPathes(path, frame):
    """
    input:
        pathlib path
        frame
    return path of calibration, path of image, path of label, path of lidar
    """
    calib = path/'calib'/f'{frame:06}.txt'
    image = path/'image_2'/f'{frame:06}.png'
    label = path/'label_2'/f'{frame:06}.txt'
    lidar = path/'velodyne'/f'{frame:06}.bin'
    return calib, image, label, lidar


def getImageFromPath(path):
    return Image.open(path)


def getLidarFromPath(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def getCalibrationFromPath(path):
    calibration = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            if ':' in line:
                key, value = line.split(':', 1)
                calibration[key] = np.array([float(x) for x in value.split()])
    return calibration


def getLabelFromPath(path):
    label = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            if len(line) > 3:
                key, value = line.split(' ', 1)
                if key in label.keys():
                    label[key].append([float(x) for x in value.split()])
                else:
                    label[key] = [[float(x) for x in value.split()]]
    return label


def getDataFromPath(path, frame):
    """
    input:
        pathlib path
        frame
    return data of calibration, data of image, data of label, data of lidar
    """
    calib, image, label, lidar = getDataPathes(path, frame)
    return getImageFromPath(image), getLidarFromPath(lidar), getCalibrationFromPath(calib), getLabelFromPath(label)


def calibDataToMatrix(calib):
    # Velodyne to/from referenece camera (0) matrix
    Tr_velo_to_cam = np.zeros((4, 4))
    Tr_velo_to_cam[3, 3] = 1
    Tr_velo_to_cam[:3, :4] = calib['Tr_velo_to_cam'].reshape(3, 4)
    R0_rect = np.zeros((4, 4))
    R0_rect[:3, :3] = calib['R0_rect'].reshape(3, 3)
    R0_rect[3, 3] = 1
    P2_rect = calib['P2'].reshape(3, 4)
    return Tr_velo_to_cam, R0_rect, P2_rect


def getTrCamToVelo(calib):
    Tr_velo_to_cam = np.zeros((4, 4))
    Tr_velo_to_cam[3, 3] = 1
    Tr_velo_to_cam[:3, :4] = calib['Tr_velo_to_cam'].reshape(3, 4)
    Tr_cam_to_velo = np.linalg.inv(Tr_velo_to_cam)
    return Tr_cam_to_velo


def applyTransformationXYZ(xyz, calib):
    """
    xyz size: (3,1)
    """
    Tr_cam_to_velo = getTrCamToVelo(calib)
    xyz1 = np.vstack((xyz, np.ones(xyz.shape[-1])))
    return Tr_cam_to_velo.dot(xyz1)[:3]


def parseLabelData(label, calib):
    for key in label.keys():
        for idx in range(len(label[key])):
            data = {}

            # attributes
            data['truncated'] = label[key][idx][0]
            data['occluded'] = label[key][idx][1]
            data['alpha'] = label[key][idx][2]

            # 2D bbox
            data['left'] = label[key][idx][3]
            data['top'] = label[key][idx][4]
            data['right'] = label[key][idx][5]
            data['bottom'] = label[key][idx][6]

            # get 3D dimension, heading and location for camera coordinate
            w = label[key][idx][7]
            h = label[key][idx][8]
            l = label[key][idx][9]
            x = label[key][idx][10]
            y = label[key][idx][11]
            z = label[key][idx][12]
            ry = label[key][idx][13]

            # swap height, width for lidar coordinate
            w, h = h, w

            # swap length and width for lidar coordinate
            w, l = l, w

            # transform center point
            x, y, z = applyTransformationXYZ(
                np.array([x, y, z]).reshape(3, 1), calib)
            # get yaw angle of lidar coordinate
            mat = Rotation.from_euler('y', ry, degrees=False).as_matrix()
            Tr_cam_to_velo = getTrCamToVelo(calib)
            yaw = Rotation.from_matrix(
                Tr_cam_to_velo[:3, :3].dot(mat)).as_euler('xyz', degrees=False)[-1]

            # 3D dimension, heading and location for lidar coordinate
            data['w'] = w
            data['h'] = h
            data['l'] = l
            data['x'] = x
            data['y'] = y
            data['z'] = z + h/2
            data['yaw'] = yaw

            label[key][idx] = data
