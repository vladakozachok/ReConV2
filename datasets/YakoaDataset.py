import os
import numpy as np
import warnings
import pickle
import tempfile 

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

        
        self.classes = {}
        collection_ids = {}

        assert self.subset in ['train', 'test']
        collection_ids['train'] = self.read_text_file('train.txt')
        collection_ids['test'] = self.read_text_file('validation.txt')

        
        collection_names = [x.split('/')[1] for x in collection_ids[split]]
        self.classes = {name: idx for idx, name in enumerate(sorted(set(collection_names)))}

        self.datapath = [(collection_names[i], os.path.join("3d-assets-augmentation-with-docker", collection_ids[self.subset][i]) + '_normalized'+'.pt')
                         for i in range(len(collection_ids[self.subset]))]
        
        print_log(f'The size of {self.subset} data is {len(self.datapath)}', logger='PointDetect3D')
        
        

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
      # Retrieve the points and label for the given index directly as they are already tensors
      fn = self.datapath[index]
      cls = self.classes[fn[0]]
      label = np.array([cls]).astype(np.int32)
      name, _ = fn[0].split('-', 1)
    #   print(f"index : {index}")
    #   print(f"class : {cls}")
    #   print(f"label : {label}")

      try:
        point_set = self.download_and_extract_dat_file(fn[1])  # Download when needed
      except Exception as e:
        print_log(f"Failed to read data from {fn[1]}: {e}")
        point_set = torch.zeros((8192, 3))
      return 'PointDetect3D', 'sample', (point_set, label[0], name)

    def read_text_file(self, file_path):
        blob = self.bucket.blob(file_path)
        return blob.download_as_text().splitlines()

        
    def download_and_extract_dat_file(self, blob_name):
        blob = self.bucket.blob(blob_name)
        _, extension = os.path.splitext(blob_name)

        # Use a temporary file to download the file from GCP
        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temp_file:
            temp_file_path = temp_file.name  # Store the temporary file path
            
            # Download the file to the temporary location
            blob.download_to_filename(temp_file_path)
            
        try:
            # Load the tensor file
            points_tensor = torch.load(temp_file_path)
            if not isinstance(points_tensor, torch.Tensor):
                raise ValueError("Loaded file is not a PyTorch tensor.")

        except Exception as e:
            print(f"Error reading the file {temp_file_path}: {e}")
            # points_tensor = torch.tensor([])  # Return an empty tensor in case of error

        finally:
            # Clean up the temporary file
            os.remove(temp_file_path)

        return points_tensor