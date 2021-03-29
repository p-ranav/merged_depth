import argparse
import cv2
import os
import numpy as np
from .infer import InferenceEngine
from tqdm import tqdm
import torch

def main(path, delete_original):

  inputs = []
  for root, dirs, files in os.walk(path, topdown=False):
    for file in files:
      if (not file.startswith(".")) and (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg")):
        inputs.append(os.path.join(root, file))

  with torch.no_grad():
    engine = InferenceEngine()
    for path in tqdm(inputs):
      try:
        dirname = os.path.dirname(path)
        base = os.path.basename(path)
        filename_minus_ext, ext = os.path.splitext(base)

        image, depth, colorized_depth = engine.predict_depth(path)

        # Save numpy array of color+depth values
        save_path = os.path.join(dirname, filename_minus_ext + ".npz")
        np.savez_compressed(save_path, color=image, depth=depth)

        if delete_original:
          # Remove original PNG image
          os.remove(path)
      except Exception as e:
        print("Exception thrown while processing image", path)
        print(e)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Run the depth predictor recursively on all .jpeg/.png files in directory (ideal for datasets)')
  parser.add_argument('path', metavar='path', type=str, help='path to dataset')
  parser.add_argument('-d', '--delete-original', default=False, action='store_true', help='Delete original .jpeg/.png after inference (default=False)')
  args = parser.parse_args()
  main(args.path, args.delete_original)
