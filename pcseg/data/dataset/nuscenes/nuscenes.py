import os
import numpy as np
from torch.utils import data
import random
import yaml
import pickle
from .LaserMix_nusc import lasermix_aug
from .PolarMix_nusc import polarmix


# used for polarmix
instance_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Omega = [np.random.random() * np.pi * 2 / 3, (np.random.random() + 1) * np.pi * 2 / 3]


class NuscenesDataset(data.Dataset):
    def __init__(self,
                 data_cfgs,
                 root_path,
                 training,
                 logger = None,
                 use_scene_flow = False
                 ):
        super().__init__()
        self.data_cfgs = data_cfgs
        self.root_path = root_path
        self.training = training
        self.logger = logger
        self.tta = data_cfgs.get('TTA', False)
        self.train_val = data_cfgs.get('TRAINVAL', False)
        self.augment = data_cfgs.AUGMENT
        self.use_scene_flow = use_scene_flow
        self.split = data_cfgs.SPLIT
            
        if self.split == 'train':
            self.info_path_list = ['nuscenes_seg_infos_1sweeps_train.pkl']
        elif self.split == 'val':
            self.info_path_list = ['nuscenes_seg_infos_1sweeps_val.pkl']
        elif self.split == 'train_val':
            self.info_path_list = ['nuscenes_seg_infos_1sweeps_train.pkl', 'nuscenes_seg_infos_1sweeps_val.pkl']
        elif self.split == 'test':
            self.info_path_list = ['nuscenes_seg_infos_1sweeps_test.pkl']
        else:
            raise Exception('split must be train/val/train_val/test.')
        
        with open("pcseg/data/dataset/nuscenes/nuscenes.yaml", 'r') as stream:
            self.nuscenes_dict = yaml.safe_load(stream)

        self.infos = []   # 每一个元素是一个dict
        for info_path in self.info_path_list:
            with open(os.path.join(self.root_path, info_path), 'rb') as f:
                infos = pickle.load(f)
                self.infos.extend(infos)
        
        self.infos_another = self.infos.copy()
        random.shuffle(self.infos_another)
        print(f'The total sample is {len(self.infos)}')
        
        self._sample_idx = np.arange(len(self.infos))

        self.samples_per_epoch = self.data_cfgs.get('SAMPLES_PER_EPOCH', -1)
        if self.samples_per_epoch == -1 or not self.training:
            self.samples_per_epoch = len(self.infos)

        if self.training:
            self.resample()
        else:
            self.sample_idx = self._sample_idx
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.infos)
    
    def resample(self):
        self.sample_idx = np.random.choice(self._sample_idx, self.samples_per_epoch)
    
    def __getitem__(self, index):
        info = self.infos[index]
        lidar_path = os.path.join(self.root_path, info['lidar_path'])
        raw_data = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
        raw_data = raw_data[:, :-1]   # 最后一维的时间戳信息不当作特征
        if self.split != 'test':
            lidarseg_label_path = os.path.join(self.root_path, info['lidarseg_label_path'])
            annotated_data = np.fromfile(str(lidarseg_label_path), dtype=np.uint8, count=-1).reshape([-1])
            annotated_data = np.vectorize(self.nuscenes_dict['learning_map'].get)(annotated_data).astype(np.uint8)
        else:
            annotated_data = np.zeros(raw_data.shape[0]).astype(np.uint8)
            
        # load scene flow
        if self.use_scene_flow:
            try:
                flow_data = np.load(lidar_path.replace('samples', 'samples_scene_flow')[:-7] + 'npy')
            except:
                flow_data = np.zeros((len(raw_data), 3), dtype=np.float32)
            raw_data = np.concatenate((raw_data, flow_data), axis=1)   # (n, 7)
        
        # lasermix or polarmix
        prob = np.random.choice(2, 1)
        if self.augment == 'GlobalAugment_LP' and ('train' in self.split):
            info_2 = self.infos_another[index]
            lidar_path_2 = os.path.join(self.root_path, info_2['lidar_path'])
            raw_data_2 = np.fromfile(str(lidar_path_2), dtype=np.float32, count=-1).reshape([-1, 5])
            raw_data_2 = raw_data_2[:, :-1]   # 最后一维的时间戳信息不当作特征
            
            lidarseg_label_path_2 = os.path.join(self.root_path, info_2['lidarseg_label_path'])
            annotated_data_2 = np.fromfile(str(lidarseg_label_path_2), dtype=np.uint8, count=-1).reshape([-1])
            annotated_data_2 = np.vectorize(self.nuscenes_dict['learning_map'].get)(annotated_data_2).astype(np.uint8)
                
            # load scene flow
            if self.use_scene_flow:
                try:
                    flow_data_2 = np.load(lidar_path_2.replace('samples', 'samples_scene_flow')[:-7] + 'npy')
                except:
                    flow_data_2 = np.zeros((len(raw_data_2), 3), dtype=np.float32)
                raw_data_2 = np.concatenate((raw_data_2, flow_data_2), axis=1)   # (n', 7)
            
            if prob == 1:   # laser mix
                annotated_data = annotated_data.reshape((-1, 1))
                annotated_data_2 = annotated_data_2.reshape((-1, 1))
                raw_data, annotated_data = lasermix_aug(
                    raw_data,
                    annotated_data,
                    raw_data_2,
                    annotated_data_2,
                )
            elif prob == 0:
                alpha = (np.random.random() - 1) * np.pi
                beta = alpha + np.pi
                raw_data, annotated_data = polarmix(
                    raw_data, annotated_data, raw_data_2, annotated_data_2,
                    alpha=alpha, beta=beta,
                    instance_classes=instance_classes, Omega=Omega
                )
            annotated_data = annotated_data.reshape(-1)
        
        pc_data = {
            'xyzret': raw_data,
            'labels': annotated_data.astype(np.uint8),
            'path': info['token'],
        }

        return pc_data
