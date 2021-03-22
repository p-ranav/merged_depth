import argparse
import cv2
import os
import numpy as np
from .infer import InferenceEngine

def main(path):

  def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

  engine = InferenceEngine()

  dirname = os.path.dirname(path)
  base = os.path.basename(path)
  filename_minus_ext, ext = os.path.splitext(base)

  print("Predicting depth for image", path)

  image, depth, colorized_depth = engine.predict_depth(path)

  # Save numpy array of depth values
  with open(os.path.join(dirname, filename_minus_ext + "_depth.npy"), 'wb') as file:
    np.save(file, depth)

  # Save stacked colorized depth result
  display = np.vstack([image, colorized_depth])
  cv2.imwrite(os.path.join(dirname, filename_minus_ext + "_stacked" + ext), display)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Run the depth predictor on input image')
  parser.add_argument('path', metavar='path', type=str,
                      help='path to input image')
  args = parser.parse_args()
  main(args.path)
