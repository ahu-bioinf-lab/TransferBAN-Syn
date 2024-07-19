# Interpretable bilinear attention network with domain adaptation improves drug-target prediction | [Paper](https://doi.org/10.1038/s42256-022-00605-1)

<div align="left">
</div>

## Introduction
This repository contains the PyTorch implementation of **TranferBAN-Syn** framework, as described in our paper "TransferBAN-Syn: A Transfer Learning-Based Algorithm for Predicting Synergistic Drug Combinations Against Echinococcosis". 
## Framework
![TranferBAN-Syn](image/DrugBAN.jpg)
## System Requirements
The source code developed in Python 3.8 using PyTorch 1.7.1. The required python dependencies are given below. DrugBAN is supported for any standard computer and operating system (Windows/macOS/Linux) with enough RAM to run. There is no additional non-standard hardware requirements.

```
torch>=1.7.1
dgl>=0.7.1
dgllife>=0.2.8
numpy>=1.20.2
scikit-learn>=0.24.2
pandas>=1.2.4
prettytable>=2.2.1
rdkit~=2021.03.2
yacs~=0.1.8
comet-ml~=3.23.1 # optional
```
## Installation Guide
Clone this Github repo and set up a new conda environment. It normally takes about 10 minutes to install on a normal desktop computer.
```
# create a new conda environment
$ conda create --name drugban python=3.8
$ conda activate TransferBAN-Syn

# install requried python dependencies
$ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
$ conda install -c dglteam dgl-cuda10.2==0.7.1
$ conda install -c conda-forge rdkit==2021.03.2
$ pip install dgllife==0.2.8
$ pip install -U scikit-learn
$ pip install yacs
$ pip install prettytable

# clone the source code of DrugBAN
$ git clone https://github.com/pz-white/DrugBAN.git
$ cd TransferBAN-Syn
```


