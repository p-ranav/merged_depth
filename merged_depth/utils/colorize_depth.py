import matplotlib
import matplotlib.cm
import cv2

def colorize_depth(arr, vmin=None, vmax=None, cmap='magma_r', ignore=-1):
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

  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  return img
