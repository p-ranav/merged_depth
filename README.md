<p align="center">
  <img height="90" src=".logo/logo.png"/>  
</p>

`merged_depth` runs (1) [AdaBins](https://github.com/shariqfarooq123/AdaBins) (NYU + KITTI models), (2) [DiverseDepth](https://github.com/YvanYin/DiverseDepth), (3) [MiDaS](https://github.com/intel-isl/MiDaS), (4) [SGDepth](https://github.com/ifnspaml/SGDepth), and (5) [Monodepth2](https://github.com/nianticlabs/monodepth2), and calculates the average predicted depth.

## Quick Start

First, download the pretrained models using the `download_models` script. Next, run the `infer` script - this will run on all images in `test/input` and save the results to `test/output`. You can use [`InferenceEngine.predict_depth(image)`](https://github.com/p-ranav/merged_depth/blob/master/merged_depth/infer.py#L335) if you just want to run the inference on a single image

```console
$ python3 -m pip install -r requirements.txt
$ python3 -m merged_depth.utils.download_models
$ python3 -m merged_depth.infer
```

## Sample Output

| <!-- -->                       | <!-- -->                        | <!-- -->                        | <!-- -->                        |
:-------------------------------:|:-------------------------------:|:-------------------------------:|:-------------------------------:|
![](./test/output/00_depth.png)  | ![](./test/output/01_depth.png) | ![](./test/output/05_depth.png) | ![](./test/output/06_depth.png) |
![](./test/output/07_depth.png)  | ![](./test/output/08_depth.png) | ![](./test/output/10_depth.png) | ![](./test/output/12_depth.png) |
![](./test/output/13_depth.png)  | ![](./test/output/16_depth.png) | ![](./test/output/17_depth.png) | ![](./test/output/18_depth.png) |
![](./test/output/23_depth.png)  | ![](./test/output/20_depth.png) | ![](./test/output/25_depth.png) | ![](./test/output/27_depth.png) |
![](./test/output/28_depth.png)  | ![](./test/output/29_depth.png) | ![](./test/output/30_depth.png) | ![](./test/output/31_depth.png) |
![](./test/output/32_depth.png)  | ![](./test/output/33_depth.png) | ![](./test/output/34_depth.png) | ![](./test/output/36_depth.png) |
![](./test/output/37_depth.png)  | ![](./test/output/39_depth.png) | ![](./test/output/40_depth.png) | ![](./test/output/42_depth.png) |
![](./test/output/43_depth.png)  | ![](./test/output/45_depth.png) | ![](./test/output/47_depth.png) | ![](./test/output/49_depth.png) |
