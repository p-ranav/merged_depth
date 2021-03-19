import cv2
from PIL import Image
from infer import InferenceHelper

infer_helper = InferenceHelper(dataset='nyu', device='cpu')

img_name = '10015'

# predict depth of a single pillow image
img = Image.open("/Users/pranav/Desktop/example_frames/" + img_name + ".png")  # any rgb pillow image
bin_centers, predicted_depth = infer_helper.predict_pil(img)

predicted_depth = predicted_depth.squeeze()
print(predicted_depth.shape)
print(predicted_depth)

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

cv2.imshow('depth', colorize(predicted_depth))
cv2.waitKey(0)
cv2.destroyAllWindows()