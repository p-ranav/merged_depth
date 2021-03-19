from tools.parse_arg_test import TestOptions
from data.load_dataset import CustomerDataLoader
from lib.models.diverse_depth_model import RelDepthModel
from lib.utils.net_tools import load_ckpt
import torch
import os
import numpy as np
from lib.core.config import cfg, merge_cfg_from_file
import matplotlib.pyplot as plt
from lib.utils.logging import setup_logging, SmoothedValue

import torchvision.transforms as transforms
from lib.utils.evaluate_depth_error import evaluate_rel_err, recover_metric_depth
logger = setup_logging(__name__)
import cv2
import json
from PIL import Image

def scale_torch(img, scale):
    """
    Scale the image and output it in torch.tensor.
    :param img: input image. [C, H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    img = np.transpose(img, (2, 0, 1))
    img = img[::-1, :, :]
    img = img.astype(np.float32)
    img /= scale
    img = torch.from_numpy(img.copy())
    img = transforms.Normalize(cfg.DATASET.RGB_PIXEL_MEANS, cfg.DATASET.RGB_PIXEL_VARS)(img)
    return img

import matplotlib
def colorize(arr, vmin=0.1, vmax=6, cmap='plasma_r', ignore=-1):
    invalid_mask = arr == ignore

    # normalize
    vmin = arr.min() if vmin is None else vmin
    vmax = arr.max() if vmax is None else vmax
    if vmin != vmax:
        arr = (arr - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        arr = arr * 0.
    cmapper = matplotlib.cm.get_cmap(cmap)
    arr = cmapper(arr, bytes=True)  # (nxmx4)
    arr[invalid_mask] = 255
    img = arr[:, :, :3]

    return img

if __name__ == '__main__':
    test_args = TestOptions().parse()
    test_args.thread = 1
    test_args.batchsize = 1
    merge_cfg_from_file(test_args)

    # load model
    model = RelDepthModel()

    model.eval()

    # load checkpoint
    if test_args.load_ckpt:
        load_ckpt(test_args, model)

    # model.cuda()
    model = torch.nn.DataParallel(model)

    for i in range(14):

      img_name = None
      if i < 10:
        img_name = '0' + str(i)
      else:
        img_name = str(i)

      # predict depth of a single pillow image
      img = Image.open("./example_frames/" + img_name + ".png")  # any rgb pillow image

      img_torch = scale_torch(img, 255)
      img_torch = img_torch[np.newaxis, :]
      print(img_torch.shape)
      pred_depth, _ = model.module.depth_model(img_torch)

      predicted_depth = pred_depth.detach().numpy()

      predicted_depth = predicted_depth.squeeze()

      print(predicted_depth.shape)
      print(predicted_depth)

      cv2.imshow('depth', colorize(predicted_depth))
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break