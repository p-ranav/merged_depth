# General imports
import argparse
import cv2
import numpy as np
import ntpath

# Helper functions
from merged_depth.utils.colorize_depth import colorize_depth

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def main(path):
    depth = np.load(path)
    window_name = path_leaf(path)
    while True:
        cv2.imshow(window_name, colorize_depth(depth))
        key = cv2.waitKey(0)
        # Shuts down the display window and terminates
        # the Python process when a specific key is
        # pressed on the window.
        # 27 is the esc key
        # 113 is the letter 'q'
        if key == 27 or key == 113:
            break
        elif cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) <1:
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Load depth numpy array from .npy file')
  parser.add_argument('path', metavar='path', type=str,
                      help='path to _depth.npy file')
  args = parser.parse_args()
  main(args.path)