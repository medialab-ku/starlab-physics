# Digital human clothing project

## Introduction
A project to dress a human character mesh created with RGB video using physics simulation.

## Requirements (Recommended)
- Python 3.8
- Ubuntu 20.04
- CUDA 12
- PyTorch 2.1.0
- Taichi 1.6.0

For other settings, please refer to the README.md file.


## Problem
The current human character mesh from deep learning model is not stable enough to be used in physics simulation.
There are two major issues currently being considered:
1. Zittering

In the case of zittering, the tremor is transmitted directly to the clothes, resulting in unstable movement.
As impacts are steadily applied to cloth, friction decreases and eventually causes clothes to gradually peel off.

2. Occluded body parts

If part of the body is obscured in the video, this cannot be handled perfectly due to the nature of the model, which only uses RGB data.
The deep learning model ultimately creates large, non-continuous movements of the character between frames in the occluded areas, which causes problems in finding collision pairs in physics simulation.


These issues cause characters to be unable to wear clothes reliably. 