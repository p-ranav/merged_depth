# General imports
import sys
import cv2
from PIL import Image
import numpy as np

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
from merged_depth.nets.AdaBins.infer import InferenceHelper

class InferenceEngine:
  def __init__(self, device='cpu'):
    self.adabins_nyu_infer_helper = InferenceHelper(dataset='nyu', device=device)
    self.adabins_kitti_infer_helper = InferenceHelper(dataset='kitti', device=device)

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

  def predict_depth(self, path):
    image = Image.open(path)
    original = cv2.imread(path)

    # Predict with AdaBins pre-trained models
    adabins_nyu_predicted_depth = self.adabins_nyu_predict(image)
    adabins_kitti_predicted_depth = self.adabins_kitti_predict(image)

    average_depth = (adabins_nyu_predicted_depth + adabins_kitti_predicted_depth) / 2

    display = np.vstack([original, colorize_depth(average_depth)])

    cv2.imwrite("output.png", display)

def main():
  engine = InferenceEngine()
  engine.predict_depth("./test/input/00.png")

if __name__ == '__main__':
  main()