import argparse
import cv2
import os
import numpy as np
from .infer import InferenceEngine
from os import listdir
from os.path import isfile, join
import torch

def main(path):

  files = [os.path.join(path, f) for f in listdir(path) if isfile(join(path, f))]

  def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

  with torch.no_grad():
    engine = InferenceEngine()

    for input_image in files:
      if (".png" in input_image or ".jpg" in input_image or ".jpeg" in input_image) and ("stacked_" not in input_image):
        dirname = os.path.dirname(input_image)
        base = os.path.basename(input_image)
        filename_minus_ext, ext = os.path.splitext(base)

        print("Predicting depth for image", input_image)

        image, depth, colorized_depth = engine.predict_depth(input_image)

        # Save numpy array of depth values
        with open(os.path.join(dirname, filename_minus_ext + "_depth.npy"), 'wb') as file:
          np.save(file, depth)

        # Save stacked colorized depth result
        display = np.vstack([image, colorized_depth])
        cv2.imwrite(os.path.join(dirname, filename_minus_ext + "_stacked" + ext), display)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Run the depth predictor on a directory of images')
  parser.add_argument('path', metavar='path', type=str,
                      help='path to directory containing images')
  args = parser.parse_args()
  main(args.path)