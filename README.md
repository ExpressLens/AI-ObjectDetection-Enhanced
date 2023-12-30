# Active Perception Model for Autonomous Driving

![](images/thumbnail.gif)

This repository hosts the official ExpressLens enhanced code of the paper:

> Active Perception using Light Curtains for Autonomous Driving. ECCV 2020.

## Quick Start

1. Clone this repository:
```bash
git clone git@github.com:ExpressLens/AI-ObjectDetection-Enhanced.git
```

2. Complete the installation steps.

3. Download the required datasets.

## Installation

Follow the installation steps in order:

1. Clone this repository.
2. Install `pylc`.
3. Install [spconv](https://github.com/traveller59/spconv).
4. Add required paths to the `$PYTHONPATH`.

## Data Preparation

Download and create info files for [Virtual KITTI](https://europe.naverlabs.com/research/computer-vision-research-naver-labs-europe/proxy-virtual-worlds-vkitti-1/) and [SYNTHIA-AL](https://synthia-dataset.net/downloads/) datasets.

## Training

To train a model, run the training commands.

## Evaluation

To evaluate a model, run the evaluation commands.

Launch all experiments on `slurm`

For reproducibility, we created a script to launch all experiments using [slurm](https://slurm.schedmd.com/documentation.html).

## Notes

- This repo uses Python 3.7 and is built upon the [SECOND](https://github.com/traveller59/second.pytorch) repository.
