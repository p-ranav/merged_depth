# General imports
import os
import sys
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.transforms import Compose
import time
import ntpath

# Fix for MacOS OpenMP error:
# OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.
# OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

sys.path.append("merged_depth/nets/AdaBins")
sys.path.append("merged_depth/nets/DiverseDepth")
sys.path.append("merged_depth/nets/MiDaS")
sys.path.append("merged_depth/nets/SGDepth")
sys.path.append("merged_depth/nets/monodepth2")

# Helper functions
from merged_depth.utils.colorize_depth import colorize_depth

# AdaBins imports
from infer import InferenceHelper

# DiverseDepth imports
from merged_depth.utils.diverse_depth_scale_image import scale_torch
from lib.models.diverse_depth_model import RelDepthModel
from lib.utils.net_tools import load_ckpt
from lib.core.config import cfg, merge_cfg_from_file
from lib.utils.logging import setup_logging, SmoothedValue
from lib.utils.evaluate_depth_error import evaluate_rel_err, recover_metric_depth

# MiDaS imports
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet

# SGDepth imports
from merged_depth.nets.SGDepth.models.sgdepth import SGDepth

# Monodepth2 imports
import networks
from layers import disp_to_depth

class InferenceEngine:
  def __init__(self):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup AdaBins model
    self.adabins_nyu_infer_helper = InferenceHelper(dataset='nyu', device=self.device)

    # Setup DiverseDepth model
    class DiverseDepthArgs:
      def __init__(self):
        self.resume = False
        self.cfg_file = "lib/configs/resnext50_32x4d_diversedepth_regression_vircam"
        self.load_ckpt = "pretrained/DiverseDepth.pth"
    diverse_depth_args = DiverseDepthArgs()
    merge_cfg_from_file(diverse_depth_args)
    self.diverse_depth_model = RelDepthModel()
    self.diverse_depth_model.eval()
    # load checkpoint
    load_ckpt(diverse_depth_args, self.diverse_depth_model)
    # TODO: update this - see how `device` argument should be processsed
    if self.device == "cuda":
      self.diverse_depth_model.cuda()
    self.diverse_depth_model = torch.nn.DataParallel(self.diverse_depth_model)

    # Setup MiDaS model
    self.midas_model_path = "./pretrained/MiDaS_f6b98070.pt"
    midas_model_type = "large"

    # load network
    if midas_model_type == "large":
      self.midas_model = MidasNet(self.midas_model_path, non_negative=True)
      self.midas_net_w, self.midas_net_h = 384, 384
    elif midas_model_type == "small":
      self.midas_model = MidasNet_small(self.midas_model_path, features=64, backbone="efficientnet_lite3",
        exportable=True, non_negative=True, blocks={'expand': True})
      self.midas_net_w, self.midas_net_h = 256, 256

    self.midas_transform = Compose(
      [
        Resize(
          self.midas_net_w,
          self.midas_net_h,
          resize_target=None,
          keep_aspect_ratio=True,
          ensure_multiple_of=32,
          resize_method="upper_bound",
          image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
      ]
    )

    self.midas_model.eval()

    self.midas_optimize = True
    if self.midas_optimize == True:
      rand_example = torch.rand(1, 3, self.midas_net_h, self.midas_net_w)
      self.midas_model(rand_example)
      traced_script_module = torch.jit.trace(self.midas_model, rand_example)
      self.midas_model = traced_script_module

      if self.device == "cuda":
        self.midas_model = self.midas_model.to(memory_format=torch.channels_last)
        self.midas_model = self.midas_model.half()

    self.midas_model.to(torch.device(self.device))

    # Setup SGDepth model
    self.sgdepth_model = InferenceEngine.SgDepthInference()

    # Setup monodepth2 model
    self.monodepth2_model_path = "pretrained/monodepth2_mono+stereo_640x192"
    monodepth2_device = torch.device(self.device)
    encoder_path = os.path.join(self.monodepth2_model_path, "encoder.pth")
    depth_decoder_path = os.path.join(self.monodepth2_model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading Monodepth2 pretrained encoder")
    self.monodepth2_encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=monodepth2_device)

    # extract the height and width of image that this model was trained with
    self.feed_height = loaded_dict_enc['height']
    self.feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.monodepth2_encoder.state_dict()}
    self.monodepth2_encoder.load_state_dict(filtered_dict_enc)
    self.monodepth2_encoder.to(monodepth2_device)
    self.monodepth2_encoder.eval()

    print("   Loading pretrained decoder")
    self.monodepth2_depth_decoder = networks.DepthDecoder(
        num_ch_enc=self.monodepth2_encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=monodepth2_device)
    self.monodepth2_depth_decoder.load_state_dict(loaded_dict)

    self.monodepth2_depth_decoder.to(monodepth2_device)
    self.monodepth2_depth_decoder.eval()

  def adabins_nyu_predict(self, image):
    _, predicted_depth = self.adabins_nyu_infer_helper.predict_pil(image)
    predicted_depth = predicted_depth.squeeze()
    predicted_depth = cv2.resize(predicted_depth, (image.width, image.height))
    return predicted_depth

  def diverse_depth_predict(self, image):
    img_torch = scale_torch(image, 255)
    img_torch = img_torch[np.newaxis, :]
    predicted_depth, _ = self.diverse_depth_model.module.depth_model(img_torch)
    predicted_depth = predicted_depth.detach().cpu().numpy()
    predicted_depth = predicted_depth.squeeze()
    return predicted_depth

  def midas_predict(self, image_path):
    """Run MonoDepthNN to compute depth maps.

    Args:
      image_path (str): path to input image
    """

    def read_image(path):
      """Read image and output RGB image (0-1).

      Args:
          path (str): path to file

      Returns:
          array: RGB image (0-1)
      """
      img = cv2.imread(path)

      if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

      return img

    # get input
    img = read_image(image_path)
    img_input = self.midas_transform({"image": img})["image"]

    # compute
    with torch.no_grad():
      sample = torch.from_numpy(img_input).to(torch.device(self.device)).unsqueeze(0)
      if self.midas_optimize == True and self.device == "cuda":
        sample = sample.to(memory_format=torch.channels_last)
        sample = sample.half()
      prediction = self.midas_model.forward(sample)
      prediction = (
        torch.nn.functional.interpolate(
          prediction.unsqueeze(1),
          size=img.shape[:2],
          mode="bicubic",
          align_corners=False,
        )
        .squeeze()
        .cpu()
        .numpy()
      )

      depth_min = prediction.min()
      depth_max = prediction.max()

      prediction = (prediction - depth_min) / (depth_max - depth_min)
      # prediction *= 10

      return prediction

  class SgDepthInference:
    """Inference without harness or dataloader"""

    def __init__(self):
      self.device = "cuda" if torch.cuda.is_available() else "cpu"
      self.model_path = "./pretrained/SGDepth_full.pth"
      self.num_classes = 20
      self.depth_min = 0.1
      self.depth_max = 100
      self.all_time = []

      self.labels = (('CLS_ROAD', (128, 64, 128)),
                      ('CLS_SIDEWALK', (244, 35, 232)),
                      ('CLS_BUILDING', (70, 70, 70)),
                      ('CLS_WALL', (102, 102, 156)),
                      ('CLS_FENCE', (190, 153, 153)),
                      ('CLS_POLE', (153, 153, 153)),
                      ('CLS_TRLIGHT', (250, 170, 30)),
                      ('CLS_TRSIGN', (220, 220, 0)),
                      ('CLS_VEGT', (107, 142, 35)),
                      ('CLS_TERR', (152, 251, 152)),
                      ('CLS_SKY', (70, 130, 180)),
                      ('CLS_PERSON', (220, 20, 60)),
                      ('CLS_RIDER', (255, 0, 0)),
                      ('CLS_CAR', (0, 0, 142)),
                      ('CLS_TRUCK', (0, 0, 70)),
                      ('CLS_BUS', (0, 60, 100)),
                      ('CLS_TRAIN', (0, 80, 100)),
                      ('CLS_MCYCLE', (0, 0, 230)),
                      ('CLS_BCYCLE', (119, 11, 32)),
                      )

      self.init_model()

    def init_model(self):
      sgdepth = SGDepth

      with torch.no_grad():
        # init 'empty' model
        self.model = sgdepth(
            1,             # opt.model_split_pos
            18,            # opt.model_num_layers
            0.9,           # opt.train_depth_grad_scale
            0.1,           # opt.train_segmentation_grad_scale
            'pretrained',  # opt.train_weights_init
            4,             # opt.model_depth_resolutions
            18,            # opt.model_num_layers_pose
        )

        # load weights (copied from state manager)
        state = self.model.state_dict()
        to_load = torch.load(self.model_path)
        for (k, v) in to_load.items():
          if k not in state:
            print(f"    - WARNING: Model file contains unknown key {k} ({list(v.shape)})")

        for (k, v) in state.items():
          if k not in to_load:
            print(f"    - WARNING: Model file does not contain key {k} ({list(v.shape)})")

          else:
            state[k] = to_load[k]

        self.model.load_state_dict(state)
        self.model = self.model.eval()
        if self.device == "cuda":
          self.model.cuda()

    def load_image(self, image):
      self.image = image
      self.image_o_width, self.image_o_height = self.image.size

      resize = transforms.Resize((192, 640))
      image = resize(self.image)  # resize to argument size

      #center_crop = transforms.CenterCrop((opt.inference_crop_height, opt.inference_crop_width))
      #image = center_crop(image)  # crop to input size

      to_tensor = transforms.ToTensor()  # transform to tensor

      self.input_image = to_tensor(image)  # save tensor image to self.input_image for saving later
      image = self.normalize(self.input_image)

      image = image.unsqueeze(0).float()
      if self.device == "cuda":
        image = image.cuda()

      # simulate structure of batch:
      image_dict = {('color_aug', 0, 0): image}  # dict
      image_dict[('color', 0, 0)] = image
      image_dict['domain'] = ['cityscapes_val_seg', ]
      image_dict['purposes'] = [['segmentation', ], ['depth', ]]
      image_dict['num_classes'] = torch.tensor([self.num_classes])
      image_dict['domain_idx'] = torch.tensor(0)
      self.batch = (image_dict,)  # batch tuple

    def normalize(self, tensor):
      mean = (0.485, 0.456, 0.406)
      std = (0.229, 0.224, 0.225)

      normalize = transforms.Normalize(mean, std)
      tensor = normalize(tensor)

      return tensor

    def get_depth_meters(self, image):
      # load image and transform it in necessary batch format
      self.load_image(image)

      start = time.time()
      with torch.no_grad():
          output = self.model(self.batch) # forward pictures

      self.all_time.append(time.time() - start)
      start = 0

      disps_pred = output[0]["disp", 0] # depth results
      segs_pred = output[0]['segmentation_logits', 0] # seg results

      segs_pred = segs_pred.exp().cpu()
      segs_pred = segs_pred.numpy()  # transform preds to np array
      segs_pred = segs_pred.argmax(1)  # get the highest score for classes per pixel

      depth_pred = disps_pred

      depth_pred = np.array(depth_pred[0][0].cpu())  # depth predictions to numpy and CPU

      def scale_depth(disp):
          min_disp = 1 / self.depth_max
          max_disp = 1 / self.depth_min
          return min_disp + (max_disp - min_disp) * disp

      depth_pred = scale_depth(depth_pred)  # Depthmap in meters

      return depth_pred

  def sgdepth_predict(self, image):
    return self.sgdepth_model.get_depth_meters(image)

  def monodepth2_predict(self, input_image):
    original_width, original_height = input_image.size
    input_image = input_image.resize((self.feed_width, self.feed_height), Image.LANCZOS)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)

    device = torch.device(self.device)
    input_image = input_image.to(device)
    features = self.monodepth2_encoder(input_image)
    outputs = self.monodepth2_depth_decoder(features)

    disp = outputs[("disp", 0)]
    disp_resized = torch.nn.functional.interpolate(
        disp, (original_height, original_width), mode="bilinear", align_corners=False)

    _, predicted_depth = disp_to_depth(disp, 0.1, 10)

    predicted_depth = predicted_depth.detach().cpu().numpy()
    predicted_depth = predicted_depth.squeeze()
    predicted_depth = cv2.resize(predicted_depth, (original_width, original_height))
    return predicted_depth

  def predict_depth(self, path):
    image = Image.open(path)
    original = cv2.imread(path)

    # Predict with AdaBins pre-trained model
    adabins_nyu_prediction = self.adabins_nyu_predict(image)
    
    # Predict with DiverseDepth model
    diverse_depth_prediction = self.diverse_depth_predict(image)
    adabins_nyu_max = np.max(adabins_nyu_prediction)
    diverse_depth_prediction *= (adabins_nyu_max / np.max(diverse_depth_prediction))

    # Predict with MiDaS model
    midas_depth_prediction = self.midas_predict(path)
    midas_depth_prediction = (midas_depth_prediction - np.max(midas_depth_prediction)) * -1
    midas_depth_prediction *= (adabins_nyu_max / np.max(midas_depth_prediction))    

    # Predict with SGDepth
    sgdepth_depth_prediction = self.sgdepth_predict(image)
    sgdepth_depth_prediction = cv2.resize(sgdepth_depth_prediction, (adabins_nyu_prediction.shape[1], adabins_nyu_prediction.shape[0]))

    # Predict with monodepth2
    monodepth2_depth_prediction = self.monodepth2_predict(image)

    def print_min_max(label, d):
      print(label, "[" + str(np.min(d)) + ", " + str(np.max(d)) + "]")

    # print_min_max("Adabins", adabins_nyu_prediction)
    # print_min_max("DiverseDepth", diverse_depth_prediction)
    # print_min_max("MiDaS", midas_depth_prediction)
    # print_min_max("SGDepth", sgdepth_depth_prediction)
    # print_min_max("Monodepth2", monodepth2_depth_prediction)

    average_depth = (
      adabins_nyu_prediction +
      diverse_depth_prediction +
      midas_depth_prediction * 5 +
      sgdepth_depth_prediction +
      monodepth2_depth_prediction
    ) / 9

    # print_min_max("Average", average_depth)
    # print("---------------------------------------")

    return original, average_depth, colorize_depth(average_depth)

def main():

  def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

  engine = InferenceEngine()

  for (dirpath, dirnames, filenames) in os.walk("./test/input"):
    for file in sorted(filenames):
      if ".png" in file or ".jpeg" in file:
        print("Predicting depth for image", file)
        path = os.path.join(dirpath, file)
        filename_minus_ext, ext = os.path.splitext(file)

        # Predict depth
        image, depth, colorized_depth = engine.predict_depth(path)

        # Save numpy array of depth values
        with open(os.path.join("./test/output", filename_minus_ext + "_depth.npy"), 'wb') as file:
          np.save(file, depth)

        # Save stacked colorized depth result
        display = np.vstack([image, colorized_depth])
        cv2.imwrite(os.path.join("./test/output", filename_minus_ext + "_stacked" + ext), display)

if __name__ == '__main__':
  main()
