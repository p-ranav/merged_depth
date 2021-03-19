import matplotlib

def colorize_depth(arr, vmin=0.1, vmax=6, cmap='plasma_r', ignore=-1):
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