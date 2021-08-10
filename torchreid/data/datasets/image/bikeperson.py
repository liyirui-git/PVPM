from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
import glob
import re
import warnings

from torchreid.data.datasets import ImageDataset
from torchreid.utils import read_image
import cv2
import numpy as np

from torchvision.transforms import Resize, RandomHorizontalFlip, ToTensor, Normalize

class BikePerson(ImageDataset):
    '''Bike-Person
    Reference:
        Yuan et al. Bike-Person Re-identification: A Benchmark and A Comprehensive Evaluation. IEEE Access 2016.
    
    URL: `<https://drive.google.com/file/d/1u6906LTa2xU4fibwfqkT6cEn81fjOFJa/view>`_
    Dataset statistics:
        - identities: 4579
    '''

    _junk_pids = [0, -1]
    
    dataset_dir = 'BikePerson-700-openpose'
    dataset_url = ''

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir

        data_dir = self.data_dir
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"BikePerson-700-xxxx".')
        
        self.train_dir = osp.join(self.data_dir, "bounding_box_train")
        self.query_dir = osp.join(self.data_dir, "query")
        self.gallery_dir = osp.join(self.data_dir, "bounding_box_test")
        
        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir
        ]

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)

        super(BikePerson, self).__init__(train, query, gallery, **kwargs)
        self.load_pose = isinstance(self.transform, tuple)
        if self.load_pose:
            self.train_pose_dir = osp.join(self.data_dir, 'bounding_box_pose_train')
            self.gallery_pose_dir = osp.join(self.data_dir, 'bounding_box_pose_test')
            self.query_pose_dir = osp.join(self.data_dir, 'query_pose')
            if self.mode == 'train':
                self.pose_dir = self.train_pose_dir
            elif self.mode == 'query':
                self.pose_dir = self.query_pose_dir
            elif self.mode == 'gallery':
                self.pose_dir = self.gallery_pose_dir
            else:
                raise ValueError('Invalid mode. Got {}, but expected to be '
                                 'one of [train | query | gallery]'.format(self.mode))

    def process_dir(self, dir_path, relabel=False):

        img_paths = glob.glob(osp.join(dir_path, '*.png'))

        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue # junk images are just ignored
            assert 0 <= pid <= 4579  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1 # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))

        return data

    def __getitem__(self, index):
        img_path, pid, camid = self.data[index]
        img = read_image(img_path)

        if self.load_pose:
            img_name = '.'.join(img_path.split('/')[-1].split('.')[:-1])
            pose_pic_name = img_name + '_pose_heatmaps.png'
            pose_pic_path = os.path.join(self.pose_dir, pose_pic_name)
            pose = cv2.imread(pose_pic_path, cv2.IMREAD_GRAYSCALE)
            pose = pose.reshape((pose.shape[0], 56, -1)).transpose((0,2,1)).astype('float32')
            pose[:,:,18:] = np.abs(pose[:,:,18:]-128)
            img, pose = self.transform[1](img, pose)
            img = self.transform[0](img)
            return img, pid, camid, img_path, pose
        else:
            if self.transform is not None:
                #### this line code have some problem, so it was replaced
                # img = self.transform(img)
                #### I find it want to use a transform.Compose object
                # use print(str(self.transform)) and find out I can use this code below to run the programme
                rhf = RandomHorizontalFlip()
                tt = ToTensor()
                nml = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                resize = Resize([384, 192])
                img = nml(tt(rhf(resize(img))))
                # img = resize(img)
                # img = rhf(img)
                # img = tt(img)
                # img = nml(img)
            return img, pid, camid, img_path
