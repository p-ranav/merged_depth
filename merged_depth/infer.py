# General imports
import sys
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

# Fix for MacOS OpenMP error:
# OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.
# OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

sys.path.append("merged_depth/nets/AdaBins")
sys.path.append("merged_depth/nets/DiverseDepth")
sys.path.append("merged_depth/nets/MiDaS")
sys.path.append("merged_depth/nets/SGDepth")

# Helper functions
from merged_depth.utils.colorize_depth import colorize_depth

# AdaBins imports
from infer import InferenceHelper

# DiverseDepth imports
from merged_depth.utils.diverse_depth_scale_image import scale_torch
from lib.models.diverse_depth_model import RelDepthModel
from lib.utils.net_tools import load_ckpt
from lib.core.config import cfg, merge_cfg_from_file
from lib.utils.logging import setup_logging, SmoothedValue
from lib.utils.evaluate_depth_error import evaluate_rel_err, recover_metric_depth

class InferenceEngine:
  def __init__(self, device='cpu'):

    # Setup AdaBins models
    self.adabins_nyu_infer_helper = InferenceHelper(dataset='nyu', device=device)
    self.adabins_kitti_infer_helper = InferenceHelper(dataset='kitti', device=device)

    # Setup DiverseDepth model
    class DiverseDepthArgs:
      def __init__(self):
        self.resume = False
        self.cfg_file = "lib/configs/resnext50_32x4d_diversedepth_regression_vircam"
        self.load_ckpt = "pretrained/DiverseDepth.pth"
    diverse_depth_args = DiverseDepthArgs()
    merge_cfg_from_file(diverse_depth_args)
    self.diverse_depth_model = RelDepthModel()
    self.diverse_depth_model.eval()
    # load checkpoint
    load_ckpt(diverse_depth_args, self.diverse_depth_model)
    # TODO: update this - see how `device` argument should be processsed
    if device != 'cpu':
      self.diverse_depth_model.cuda()
    self.diverse_depth_model = torch.nn.DataParallel(self.diverse_depth_model)

  def adabins_nyu_predict(self, image):
    _, predicted_depth = self.adabins_nyu_infer_helper.predict_pil(image)
    predicted_depth = predicted_depth.squeeze()
    predicted_depth = cv2.resize(predicted_depth, (image.width, image.height))
    return predicted_depth

  def adabins_kitti_predict(self, image):
    _, predicted_depth = self.adabins_kitti_infer_helper.predict_pil(image)
    predicted_depth = predicted_depth.squeeze()
    predicted_depth = cv2.resize(predicted_depth, (image.width, image.height))
    return predicted_depth

  def diverse_depth_predict(self, image):
    img_torch = scale_torch(image, 255)
    img_torch = img_torch[np.newaxis, :]
    predicted_depth, _ = self.diverse_depth_model.module.depth_model(img_torch)
    predicted_depth = predicted_depth.detach().numpy()
    predicted_depth = predicted_depth.squeeze()
    return predicted_depth

  def predict_depth(self, path):
    image = Image.open(path)
    original = cv2.imread(path)

    # Predict with AdaBins pre-trained models
    adabins_nyu_prediction = self.adabins_nyu_predict(image)
    adabins_kitti_prediction = self.adabins_kitti_predict(image)

    # Predict with DiverseDepth model
    diverse_depth_prediction = self.diverse_depth_predict(image)
    adabins_nyu_max = np.max(adabins_nyu_prediction)
    adabins_kitti_max = np.max(adabins_kitti_prediction)
    adabins_avg_max = (adabins_nyu_max + adabins_kitti_max) / 2
    diverse_depth_max = np.max(diverse_depth_prediction)
    scale_factor = adabins_avg_max / diverse_depth_max
    diverse_depth_prediction *= scale_factor

    average_depth = (adabins_nyu_prediction + adabins_kitti_prediction + diverse_depth_prediction) / 3

    display = np.vstack([original, colorize_depth(average_depth)])

    cv2.imwrite("output.png", display)

def main():
  engine = InferenceEngine()
  engine.predict_depth("./test/input/00.png")

if __name__ == '__main__':
  main()