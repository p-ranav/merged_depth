import numpy as np
import torch
import torchvision.transforms as transforms
from lib.core.config import cfg

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
  if torch.cuda.is_available():
    img = img.to(torch.device("cuda"))
  img = transforms.Normalize(cfg.DATASET.RGB_PIXEL_MEANS, cfg.DATASET.RGB_PIXEL_VARS)(img)
  return img