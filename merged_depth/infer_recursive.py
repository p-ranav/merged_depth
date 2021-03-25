import argparse
import cv2
import os
import numpy as np
from .infer import InferenceEngine
from tqdm import tqdm

def main(path):

  inputs = []
  for root, dirs, files in os.walk(path, topdown=False):
    for file in files:
      if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
        inputs.append(os.path.join(root, file))

  engine = InferenceEngine()
  for path in tqdm(inputs):
    dirname = os.path.dirname(path)
    base = os.path.basename(path)
    filename_minus_ext, ext = os.path.splitext(base)

    image, depth, colorized_depth = engine.predict_depth(path)

    # Save numpy array of depth values
    with open(os.path.join(dirname, filename_minus_ext + "_depth.npz"), "wb") as npz:
      np.savez_compressed(npz, depth)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Run the depth predictor recursively on all .jpeg/.png files in directory (ideal for datasets)')
  parser.add_argument('path', metavar='path', type=str, help='path to dataset')
  args = parser.parse_args()
  main(args.path)
