<p align="center">
  <img height="90" src=".logo/logo.png"/>  
</p>

`merged_depth` runs (1) [AdaBins](https://github.com/shariqfarooq123/AdaBins), (2) [DiverseDepth](https://github.com/YvanYin/DiverseDepth), (3) [MiDaS](https://github.com/intel-isl/MiDaS), (4) [SGDepth](https://github.com/ifnspaml/SGDepth), and (5) [Monodepth2](https://github.com/nianticlabs/monodepth2), and calculates a weighted-average per-pixel absolute depth estimation.

## Quick Start

First, download the pretrained models using the `download_models` script. 

Next, run the `infer` script - this will run on all images in `test/input` and save the results to `test/output`. 

```bash
python3 -m pip install -r requirements.txt
python3 -m merged_depth.utils.download_models
python3 -m merged_depth.infer
```

The results include (1) a `_depth.npy` file that you can load, (2) a `_stacked.png` file that shows the original and colorized depth images. 

To run the predictor on a single input, use `infer_single.py`

```bash
python3 -m merged_depth.infer_single ~/foo/bar/test.png
```

## Sample Output

| <!-- -->                       | <!-- -->                        | <!-- -->                        | <!-- -->                        |
:-------------------------------:|:-------------------------------:|:-------------------------------:|:-------------------------------:|
![](./test/output/00_stacked.png)  | ![](./test/output/01_stacked.png) | ![](./test/output/05_stacked.png) | ![](./test/output/06_stacked.png) |
![](./test/output/07_stacked.png)  | ![](./test/output/08_stacked.png) | ![](./test/output/10_stacked.png) | ![](./test/output/12_stacked.png) |
![](./test/output/13_stacked.png)  | ![](./test/output/16_stacked.png) | ![](./test/output/17_stacked.png) | ![](./test/output/18_stacked.png) |
![](./test/output/23_stacked.png)  | ![](./test/output/20_stacked.png) | ![](./test/output/25_stacked.png) | ![](./test/output/27_stacked.png) |
![](./test/output/28_stacked.png)  | ![](./test/output/29_stacked.png) | ![](./test/output/30_stacked.png) | ![](./test/output/31_stacked.png) |
![](./test/output/32_stacked.png)  | ![](./test/output/33_stacked.png) | ![](./test/output/34_stacked.png) | ![](./test/output/36_stacked.png) |
![](./test/output/37_stacked.png)  | ![](./test/output/39_stacked.png) | ![](./test/output/40_stacked.png) | ![](./test/output/42_stacked.png) |
![](./test/output/43_stacked.png)  | ![](./test/output/45_stacked.png) | ![](./test/output/47_stacked.png) | ![](./test/output/49_stacked.png) |
