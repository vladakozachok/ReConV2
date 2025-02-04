import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset
from .build import DATASETS
from utils.logger import *
import torch
from google.cloud import storage

warnings.filterwarnings('ignore')


@DATASETS.register_module()
class YakoaDataset(Dataset):
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
        self.client = storage.Client()
        self.bucket = self.client.bucket(self.root)

        
        # Acquire all the classes
        self.catfile = 'collection_names.txt'  # classes
        self.cat = self.read_text_file(self.catfile)
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        collection_ids = {}

        collection_ids['train'] = self.read_text_file('model_train.txt')
        collection_ids['test'] = self.read_text_file('model_test.txt')

        assert (split == 'train' or split == 'test')
        
        collection_names = ['_'.join(x.split('-')[0:-1]) for x in collection_ids[split]]
        self.datapath = [(collection_names[i], os.path.join("unique-asset-clouds", collection_ids[split][i]) + '.dat') for i
                         in range(len(collection_ids[split]))]

        print_log('The size of %s data is %d' % (split, len(self.datapath)), logger='PointDetect3D')

        if self.uniform:
            self.save_path = 'pointdetect3d%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints)
        else:
            self.save_path = 'pointdetect3d%d_%s_%dpts.dat' % (self.num_category, split, self.npoints)

        if self.process_data:
            if not os.path.exists(self.save_path):
                print_log('Processing data %s (only running in the first time)...' % self.save_path, logger='PointDetect')
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

    def read_text_file(self, file_path):
        blob = self.bucket.blob(file_path)
        return blob.download_as_text().splitlines()
        
    def download_and_extract_dat_file(self, blob_name):
        print(blob_name)
        blob = self.bucket.blob(blob_name)
        _, extension = os.path.splitext(blob_name)
        
        with tempfile.NamedTemporaryFile(suffix=extension) as temp_file:
            blob.download_to_filename(temp_file.name)
            with open(temp_file.name, 'rb') as f:
                point_cloud = pickle.load(f)

            return point_cloud
