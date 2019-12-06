# VividNet

VividNet is a proof-of-concept implementation for our papers (https://arxiv.org/abs/1905.08910 and https://arxiv.org/abs/1905.09891). The goal of VividNet is to be an inverse-simulation pipeline based on neural-symbolic capsule networks. 

This project is still **work-in-progress**. At this point, the network is able to perform inverse-graphics, rendering and intuitive physics.

## Dependencies

Our implementation requires the following external modules:
  - [NumPy](https://www.numpy.org/)
  - [SciPy](https://www.scipy.org/)
  - [Matplotlib](https://matplotlib.org/)
  - [Keras](https://keras.io/)
  - [Numba](https://numba.pydata.org/)

## How-to

**NOTE1:** At this stage, VividNet is highly unoptimized. All examples run extremely slow. We believe this is mainly due to the constant context switching our algorithm has to do on the graphics card (between Keras and Numba).

**NOTE2:** Due to the model size, we did not include them in the Repo. However, VividNet trains itself on synthetic data automatically, when the models aren't present. This takes several hours. Be warned.

To run the examples, clone the repo and run either:

| Filename | Purpose |
| ------ | ------ |
| *mainA.py* | Example of detecting a space-ship |
| *mainB.py* | Example of Intuitive Physics with 3 circles |
| *mainC.py* | Example of Intuitive Physics *(unfinished)*|
| *mainD.py* | Example of Querying (Game Engine) *(unfinished)*|

