# Exploring Self-attention for Semantics Segmentation
by B. Shaashua, TAU

### Introduction

This repository is build for incorporating self-attention blocks (SA-Blocks) [by H. Zhao et al] into Deeplabv3. It contains full training and testing code. The implementation of SA module with optimized CUDA kernels are also included from: https://github.com/hszhao/SAN.git

### Usage

1. Requirement:

   - Hardware: trained & tested with NVIDIA GeForce GTX 1080 Ti.
   - Software: trained & tested with PyTorch 1.8.1, Python 3.7.1, CUDA 10.2, [CuPy](https://cupy.chainer.org/) 9.1

2. Clone the repository:

   ```shell
   git clone https://github.com/BakMicS/self_attention_for_semantic_segmentation.git
   ```
   
3. Train:

   - Download the PASCAL VOC 2012 dataset from: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

   - Train a model with: main.py

   - Plot learning curves with: plot_training_log.py (runs by default in the end of training)

4. Test:

   - Test a model with: test_model.py 
   
5. Visualization:

   - run_one_image.py

### Performance

Train Parameters: batch_size(6), epochs(50), base_lr(1e-4), lr_scheduler(step-wise-decay, gamma=0.98, step=2).

Overall result:

| Deeplabv3 Head type | mIOU  | Params |
| :-----------------: | :---: | :----: |
| Convolution ASPP    | 72.88 | 83.8M  |
| Pairwise SA ASPP    | 73.63 | 50.9M  |
| Patchwise SA ASPP   | 74.79 | 53.3M  |


### Citation

If you find the code or trained models useful, please consider citing: B. Shaashua, TAU
