# Decoupling-face-inpainting

This repository combines a facial structure estimator and an inpainting module for structure-aware face inpainting.

This code is based on the paper **"Decoupling Latent 3D Facial Structure for Largely Masked Face Image Inpainting"**.

## Overview

![Overall framework](overall.png)

## Inpainting Results

![Inpainting results](inpainting_results.png)

## Depth-to-Normal Conversion

The code for converting LAP depth maps to normal maps is available in [`depth_to_normal_lap.py`](https://github.com/uestc-970703/Decoupling-face-inpainting/blob/main/depth_to_normal_lap.py).

## Dataset Links

- CelebA-HQ: [https://github.com/tkarras/progressive_growing_of_gans#preparing-datasets-for-training](https://github.com/tkarras/progressive_growing_of_gans#preparing-datasets-for-training)
- FFHQ: [https://github.com/NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset)

## Download Links

### Facial Structure Estimator Checkpoint

- Baidu Netdisk: [https://pan.baidu.com/s/1Fmvvd_D7YgNbGSlUzXHaXw](https://pan.baidu.com/s/1Fmvvd_D7YgNbGSlUzXHaXw)
- Extraction code: `a1b2`

### Inpainting Module Checkpoints

- Baidu Netdisk: [https://pan.baidu.com/s/1U64Yo_H4WFrfNj84jdlBkw](https://pan.baidu.com/s/1U64Yo_H4WFrfNj84jdlBkw)
- Extraction code: `z777`
