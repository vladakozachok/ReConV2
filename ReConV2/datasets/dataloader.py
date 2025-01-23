import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset
from .build import DATASETS
from ReConV2.utils.logger import *
import torch

warnings.filterwarnings('ignore')


@DATASETS.register_module()
class PointDetect3D(Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.N_POINTS
        self.use_normals = config.USE_NORMALS
        self.num_category = config.NUM_CATEGORY
        self.process_data = True
        self.uniform = True
        split = config.subset
        self.subset = config.subset
        self.with_color = config.with_color

        
        self.catfile = os.path.join(self.root, 'model_train.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        contract_ids = {}

        contract_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'model_train.txt'))]
        contract_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'model_test.txt'))]

        assert (split == 'train' or split == 'test')
        contract_names = ['_'.join(x.split('_')[0:-1]) for x in contract_ids[split]]
        self.datapath = [(contract_names[i], os.path.join(self.root, contract_names[i], contract_names[split][i]) + '.txt') for i
                         in range(len(contract_names[split]))]
        print_log('The size of %s data is %d' % (split, len(self.datapath)), logger='ModelNet')

        if self.uniform:
            self.save_path = 'cached_data%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints)
        else:
            self.save_path = 'cached_data%d_%s_%dpts.dat' % (self.num_category, split, self.npoints)

        if self.process_data:
            if not os.path.exists(self.save_path):
                print_log('Processing data %s (only running in the first time)...' % self.save_path, logger='ModelNet')
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)

                    try:
                        np_points = self.download_and_extract_dat_file(fn[1])
                        points_tensor = torch.tensor(np_points, dtype=torch.float)

                    except Exception as e:
                        print(f"Failed to read data from {fn[1]}: {e}")
                        continue  # Skip this file and move on to the next

                    self.list_of_points[index] = points_tensor
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print_log('Load processed data from %s...' % self.save_path, logger='ModelNet')
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
      # Retrieve the points and label for the given index directly as they are already tensors
      point_set = self.list_of_points[index]
      label = self.list_of_labels[index]

      fn = self.datapath[index]
      cls = self.classes[self.datapath[index][0]]
      label = np.array([cls]).astype(np.int32)

    #   print(f"point_set: {point_set}")
    #   print(f"label[0]: {label[0]}")

      return 'PointDetect3D', 'sample', (point_set, label[0])

    def __getitem__(self, index):
        points, label = self._get_item(index)
        pt_idxs = np.arange(0, points.shape[0])  # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        return 'ModelNet', 'sample', (current_points, label)
