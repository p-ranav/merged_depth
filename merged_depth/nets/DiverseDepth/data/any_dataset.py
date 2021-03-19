import cv2
import json
import torch
import os.path
import numpy as np
import scipy.io as sio
from lib.core.config import cfg
import torchvision.transforms as transforms
from lib.utils.logging import setup_logging

logger = setup_logging(__name__)


import os
import pathlib
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from pathlib import Path

class ColorDataset:
  '''
  DataLoader for YouTube color dataset

  Parses a directory of .png color images extracted from YouTube videos
  The files are numbered like so: 0.png, 1.png, 2.png, ..., 1000.png, 1001.png, ...
  '''
  def __init__(self, dataset_directory):
    self.dataset_directory = dataset_directory
    self.value = list(
      [
        ColorDataset.Frame(os.path.join(self.dataset_directory, i))
        for i in sorted(Path(dataset_directory).iterdir(), key=os.path.getmtime)
      ]
    )

  class Frame:
    '''
    One frame of data in the dataset. This frame includes:
    * Path to color image
    * Path to depth image
    * Image width and height
    '''
    def __init__(self, color_image_path):
      self.color_image_path = color_image_path

    def get_color_image(self, resize=None):
      image = Image.open(self.color_image_path).convert("RGB")
      image = np.array(image)
      if resize != None:
        image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)
      return image

  def __iter__(self):
    return iter(self.value)

  def __len__(self):
    return len(self.value)

  def get_frame_at_index(self, index):
    return self.value[index]

class ANYDataset():
    def initialize(self, opt):
        dataset = ColorDataset(opt.dataroot)
        self.data_size = len(dataset)

    def __getitem__(self, anno_index):
        return dataset.get_frame_at_index(anno_index).get_color_image()

    def __len__(self):
        return self.data_size

    def name(self):
        return 'ANY'
