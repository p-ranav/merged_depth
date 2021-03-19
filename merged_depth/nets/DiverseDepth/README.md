#### DiverseDepth: Affine-invariant Depth Prediction Using Diverse Data.

This repository contains a FORK of the source code of this paper:
[Wei Yin, Xinlong Wang, Chunhua Shen, Yifan Liu, Zhi Tian, Songcen Xu, Changming Sun, DiverseDepth: Affine-invariant Depth Prediction Using Diverse Data](https://arxiv.org/abs/2002.00569).

## Quick Start

Run the DiverseDepth depth prediction on a .png input file using:

```console
$ python3 -m tools.test_diversedepth_png --dataroot /Users/pranav/Desktop/example_frames --dataset any --cfg_file lib/configs/resnext50_32x4d_diversedepth_regression_vircam --load_ckpt ./DiverseDepth.pth
INFO parse_arg_base.py:  56: ----------------- Options ---------------
                  base_lr: 1e-05
                batchsize: 2
                 cfg_file: lib/configs/resnext50_32x4d_diversedepth_regression_vircam	[default: lib/configs/resnext_32x4d_nyudv2_c1]
                 dataroot: /Users/pranav/Desktop/example_frames	[default: None]
                  dataset: any                           	[default: nyudv2_rel]
             dataset_list: None
         diff_loss_weight: 1
                    epoch: 100
                load_ckpt: ./DiverseDepth.pth            	[default: None]
               local_rank: 0
                loss_mode: SSIL_VNL
                    optim: SGD
                    phase: test
               phase_anno: test
              results_dir: ./evaluation
                   resume: False
         sample_depth_flg: False
       sample_ratio_steps: 10000
       sample_start_ratio: 0.1
         scale_decoder_lr: 1
              start_epoch: 0
               start_step: 0
                   thread: 1
              use_tfboard: False
----------------- End -------------------
/Users/pranav/Documents/Projects/DiverseDepth/lib/core/config.py:132: YAMLLoadWarning: calling yaml.load() without Loader=... is deprecated, as the default Loader is unsafe. Please read https://msg.pyyaml.org/load for full details.
  yaml_cfg = AttrDict(yaml.load(f))
INFO net_tools.py:  40: loading checkpoint ./DiverseDepth.pth
torch.Size([1, 3, 480, 854])
/Users/pranav/opt/anaconda3/envs/pyroshi/lib/python3.8/site-packages/torch/nn/functional.py:2952: UserWarning: nn.functional.upsample is deprecated. Use nn.functional.interpolate instead.
  warnings.warn("nn.functional.upsample is deprecated. Use nn.functional.interpolate instead.")
tensor([[[[0.4962, 0.4882, 0.5051,  ..., 0.9591, 0.8918, 0.9227],
          [0.4877, 0.4818, 0.4984,  ..., 0.9416, 0.8738, 0.8987],
          [0.5105, 0.5025, 0.5052,  ..., 0.9589, 0.9369, 0.9708],
          ...,
          [0.3981, 0.3958, 0.3993,  ..., 0.3272, 0.3274, 0.3282],
          [0.3881, 0.3850, 0.4023,  ..., 0.3347, 0.3287, 0.3296],
          [0.3891, 0.3856, 0.4023,  ..., 0.3351, 0.3285, 0.3301]]]],
       grad_fn=<MkldnnConvolutionBackward>)
(1, 1, 480, 854)
```

## Some Results

![Any images online](./examples/any_imgs.jpg)
![Point cloud](./examples/pcd.png)

## Some Dataset Examples
![Dataset](./examples/dataset_examples.png)


****
## Hightlights
- **Generalization:** We have tested on several zero-shot datasets to test the generalization of our method. 



****
## Installation
- Please refer to [Installation](./Installation.md).

## Datasets
We collect multiply source data to construct our DiverseDepth dataset, including crawling online stereoscopic images, images from DIML and Taskonomy. These three parts form the foreground parts (Part-fore), outdoor scenes (Part-out) and indoor scenes (Part-in) of our dataset. 
The size of three parts are:
Part-in:  contains 93838 images
Part-out: contains 120293 images
Part-fore: contains 109703 images
 We will release the dataset as soon as possible. 
  
## Model Zoo
- ResNext50_32x4d backbone, trained on DiverseDepth dataset, download [here](https://cloudstor.aarnet.edu.au/plus/s/ixWf3nTJFZ0YE4q)


  
## Inference

```bash
# Run the inferece on NYUDV2 dataset
 python  ./tools/test_diversedepth_nyu.py \
		--dataroot    ./datasets/NYUDV2 \
		--dataset     nyudv2 \
		--cfg_file     lib/configs/resnext50_32x4d_diversedepth_regression_vircam \
		--load_ckpt   ./model.pth 
		
# Test depth predictions on any images, please replace the data dir in test_any_images.py
 python  ./tools/test_any_diversedepth.py \
		--dataroot    ./ \
		--dataset     any \
		--cfg_file     lib/configs/resnext50_32x4d_diversedepth_regression_vircam \
		--load_ckpt   ./model.pth 
```
If you want to test the kitti dataset, please see [here](./datasets/KITTI/README.md)



### Citation
```
@article{yin2020diversedepth,
  title={DiverseDepth: Affine-invariant Depth Prediction Using Diverse Data},
  author={Yin, Wei and Wang, Xinlong and Shen, Chunhua and Liu, Yifan and Tian, Zhi and Xu, Songcen and Sun, Changming and Renyin, Dou},
  journal={arXiv preprint arXiv:2002.00569},
  year={2020}
}
```
### Contact
Wei Yin: wei.yin@adelaide.edu.au
