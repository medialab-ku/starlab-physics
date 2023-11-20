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
This causes the clothes to dig into the body rather than maintaining a stable collision relationship with the character.

These issues cause characters to be unable to wear clothes reliably. 

## Existing methods
- CCD (Continous Collision Detection)

CCD is a method of detecting collisions between objects by checking the trajectory of the object between frames.
Current thinking is that CCD cannot use perfect time steps, so character movement must be limited. 
However, in the current situation, this method is not appropriate because character information is determined by model output.

- Velocity correction

Velocity correction is a method of correcting the velocity of an object when a collision occurs. 
This method is used to prevent objects from penetrating each other when a collision occurs.
However, this method is not appropriate for the current situation because it is not possible to determine the collision pair between the character and the clothes.


## Solution
