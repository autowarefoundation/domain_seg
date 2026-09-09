## augmentations.py
Image augmentation is an essential part of training visual neural networks. It helps harden the network to various noise effects during training and helps to reduce data over-fitting. The [albumentations library](https://albumentations.ai/) is used to create different image augmentations effects simulation weather, debris, lens effects, noise, colour shifting, and mixing of image patches

![Augmentations Network Diagram](../../Media/Augmentations.jpg)

## check_data.py
Helper class to perform a sanity check on data for processing, ensuring that data is read and that number of ground-truth samples match the number of training images

## load_data_domain_seg.py
Helper class for the [DomainSeg Neural network](https://github.com/autowarefoundation/domain_seg) to load multiple datasets, separate data into training and validation splits and extract a Region of Interest (ROI) from images

## benchmark.py
Script to print model layers, number of parameters, and measure inference speed of model at either FP32 or FP16 precision.

### Example usage
```bash
  python3 benchmark.py -n SceneSeg -p fp32
```
### Parameters:

*-n , --name* : name of the network (SceneSeg, Scene3D)

*-p , --precision* : precision of the network at which to perform benchmarking (fp32, fp16)
