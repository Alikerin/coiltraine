# @Author: Ibrahim Salihu Yusuf <yusuf>
# @Date:   2020-09-17T07:01:39+01:00
# @Email:  sibrahim1396@gmail.com
# @Last modified by:   yusuf
# @Last modified time: 2020-09-17T07:24:50+01:00



import os
import glob
import traceback
import collections
import sys
import math
import copy
import json
import random
import numpy as np
import lmdb

import torch
import cv2

from torch.utils.data import Dataset

from . import splitter
from . import data_parser

# TODO: Warning, maybe this does not need to be included everywhere.
from configs import g_conf

from coilutils.general import sort_nicely



def parse_remove_configuration(configuration):
    """
    Turns the configuration line of sliptting into a name and a set of params.
    """

    if configuration is None:
        return "None", None
    print('conf', configuration)
    conf_dict = collections.OrderedDict(configuration)

    name = 'remove'
    for key in conf_dict.keys():
        if key != 'weights' and key != 'boost':
            name += '_'
            name += key

    return name, conf_dict


def get_episode_weather(episode):
    with open(os.path.join(episode, 'metadata.json')) as f:
        metadata = json.load(f)
    print(" WEATHER OF EPISODE ", metadata['weather'])
    return int(metadata['weather'])


class CoILDataset(Dataset):
    def __init__(self, root_dir, transform=None, preload_name=None, max_frames=None):

        self.image_names = []
        self.measurements = []
        self.txn = []
        self.full_path = []
        self.transform = transform
        self.batch_read_number = 0
        n_episodes = 0

        for full_path in sorted(glob.glob('%s/**' % root_dir), reverse=True):
            txn = lmdb.open(
                    full_path,
                    max_readers=1, readonly=True,
                    lock=False, readahead=False, meminit=False).begin(write=False)

            n = int(txn.get('len'.encode())) #- self.gap * self.n_step

            for i in range(n):
                if max_frames and len(self) >= max_frames:
                    break
                for _dir in ["leftrgb_", "rightrgb_", "centralrgb_"]:
                    self.image_names.append(_dir+"%04d" % i)
                    self.measurements.append('measurements_%04d' % i)
                    self.txn.append(txn)
                    self.full_path = full_path

            n_episodes += 1

            if max_frames and len(self) >= max_frames:
                break

        print('%s: %d frames, %d episodes.' % (root_dir, len(self), n_episodes))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        meas = self.measurements[idx]
        lmdb_txn = self.txn[idx]

        img = np.frombuffer(lmdb_txn.get(img_name.encode()), np.uint8).reshape(600,800,3)
        img = cv2.resize(img, dsize=(88, 200), interpolation=cv2.INTER_CUBIC)

        # Apply the image transformation
        if self.transform is not None:
            boost = 1
            img = self.transform(self.batch_read_number * boost, img)
        else:
            img = img.transpose(2, 0, 1)

        img = img.astype(np.float)
        img = torch.from_numpy(img).type(torch.FloatTensor)
        img = img / 255.


        measurement = np.frombuffer(lmdb_txn.get(meas.encode()), np.float32)

        ox, oy, oz, ori_ox, ori_oy, vx, vy, vz, ax, ay, az, pitch, roll, yaw, cmd, steer, throttle, brake, manual, gear  = measurement

        #calculate speed according to coiltraine/input/data_parser.py
        speed = self.get_speed(vx, vy, vz, pitch, yaw)

        #aaugment measurement according to coiltraine/input/coil_dataset.py
        if 'left' in img_name:
            #augment for -30 degrees
            steer = self.augment_measurement(steer, -30.0, speed)
        elif 'right' in img_name:
            #augment for 30 degrees
            steer = self.augment_measurement(steer, 30.0, speed)

        #return dictionary of measurement as seen in coiltraine/input/coil_dataset.py _get_final_measurement()
        measurement = {}
        measurement['position'] = torch.tensor([ox, oy, oz], dtype=torch.float32)
        measurement['orientation']= torch.tensor([ori_ox, ori_oy], dtype=torch.float32)
        measurement['velocity'] = torch.tensor([vx, vy, vz], dtype=torch.float32)
        measurement['acceleration'] = torch.tensor([ax, ay, az], dtype=torch.float32)
        # measurement['command'] = torch.tensor([cmd], dtype=torch.float32)
        measurement['steer'] = torch.tensor([steer], dtype=torch.float32)
        measurement['throttle'] = torch.tensor([throttle], dtype=torch.float32)
        measurement['brake'] = torch.tensor([brake], dtype=torch.float32)
        measurement['manual'] = torch.tensor([manual], dtype=torch.float32)
        measurement['gear'] = torch.tensor([gear], dtype=torch.float32)
        measurement['speed_module'] = torch.tensor([speed / g_conf.SPEED_FACTOR], dtype=torch.float32)
        measurement['game_time'] = torch.tensor([0.0], dtype=torch.float32)
        measurement['directions'] = torch.tensor([cmd], dtype=torch.float32)
        measurement['rgb'] = img
        self.batch_read_number += 1

        return measurement

    def augment_steering(self, camera_angle, steer, speed):
        """
            Apply the steering physical equation to augment for the lateral cameras steering
        Args:
            camera_angle: the angle of the camera
            steer: the central steering
            speed: the speed that the car is going

        Returns:
            the augmented steering

        """
        time_use = 1.0
        car_length = 6.0

        pos = camera_angle > 0.0
        neg = camera_angle <= 0.0
        # You should use the absolute value of speed
        speed = math.fabs(speed)
        rad_camera_angle = math.radians(math.fabs(camera_angle))
        val = g_conf.AUGMENT_LATERAL_STEERINGS * (
            math.atan((rad_camera_angle * car_length) / (time_use * speed + 0.05))) / 3.1415
        steer -= pos * min(val, 0.3)
        steer += neg * min(val, 0.3)

        steer = min(1.0, max(-1.0, steer))

        # print('Angle', camera_angle, ' Steer ', old_steer, ' speed ', speed, 'new steer', steer)
        return steer

    def augment_measurement(self, steer, angle, speed):
        """
            Augment the steering of a measurement dict

        """
        new_steer = self.augment_steering(angle, steer, speed)
        return new_steer

    def orientation_vector(self, pitch, yaw):
        pitch = np.deg2rad(pitch)
        yaw = np.deg2rad(yaw)
        orientation = np.array([np.cos(pitch)*np.cos(yaw), np.cos(pitch)*np.sin(yaw), np.sin(pitch)])
        return orientation


    def get_speed(self, vx, vy, vz, pitch, yaw):
        vel_np = np.array([vx, vy, vz])
        speed = np.dot(vel_np, self.orientation_vector(pitch, yaw))
        return speed

    def controls_position(self):
        return np.where(self.meta_data[:, 0] == b'control')[0][0]


    """
        Methods to interact with the dataset attributes that are used for training.
    """

    def extract_targets(self, data):
        """
        Method used to get to know which positions from the dataset are the targets
        for this experiments
        Args:
            labels: the set of all float data got from the dataset

        Returns:
            the float data that is actually targets

        Raises
            value error when the configuration set targets that didn't exist in metadata
        """
        targets_vec = []
        for target_name in g_conf.TARGETS:
            targets_vec.append(data[target_name])

        return torch.cat(targets_vec, 1)

    def extract_inputs(self, data):
        """
        Method used to get to know which positions from the dataset are the inputs
        for this experiments
        Args:
            labels: the set of all float data got from the dataset

        Returns:
            the float data that is actually targets

        Raises
            value error when the configuration set targets that didn't exist in metadata
        """
        inputs_vec = []
        for input_name in g_conf.INPUTS:
            inputs_vec.append(data[input_name])

        return torch.cat(inputs_vec, 1)

    def extract_intentions(self, data):
        """
        Method used to get to know which positions from the dataset are the inputs
        for this experiments
        Args:
            labels: the set of all float data got from the dataset

        Returns:
            the float data that is actually targets

        Raises
            value error when the configuration set targets that didn't exist in metadata
        """
        inputs_vec = []
        for input_name in g_conf.INTENTIONS:
            inputs_vec.append(data[input_name])

        return torch.cat(inputs_vec, 1)
