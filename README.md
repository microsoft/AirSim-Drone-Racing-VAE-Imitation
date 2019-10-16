# Learning Controls Using Cross-Modal Representations: Bridging Simulation and Reality for Drone Racing

![Teaser](figs/main_2_low.png)

This repository provides a code base to evaluate and train models from the paper "*Learning Controls Using Cross-Modal Representations: Bridging Simulation and Reality for Drone Racing*". The original paper and video can be found in the following links:

Paper: [https://arxiv.org/abs/1909.06993](https://arxiv.org/abs/1909.06993)

Video: [https://youtu.be/aSugOL0qI44](https://youtu.be/aSugOL0qI44)

## Recommended system
Recommended system (tested):
- Ubuntu 18.04
- Python 2.7.15

Python packages used by the example provided and their recommended version:
- tensorflow==2.0.0-beta1
- msgpack-rpc-python==0.4.1
- numpy==1.16.4
- matplotlib==2.1.1

## Preprocessing for training cross-modal representation
In order to train the cross-modal representations you need to either download the image dataset here, or generate the data yourself using Airsim.

### Downloading the data

- Download the dataset [RHD dataset v. 1.1](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html)
- Extract it.
- In the file 'create_binary_db.py', set the variable 'path_to_db' to the path of the extracted dataset.
- Run
```
python create_binary_db.py
```
- This will create a binary file in *./data/bin* according to how 'set' was configured. Keep it at 'evaluation'.

