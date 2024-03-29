# ExPort(Expert Port)

```
oooooooooooo             ooooooooo                               
 888       8              888    Y88                        o8   
 888         oooo    ooo  888    d88    ooooo   oooo d8b  o888oo 
 888oooo8      88b  8P    888ooo88P   d88   88b  888  8P   888   
 888            Y888      888         888   888  888       888   
 888       o   o8  88b    888         888   888  888       888   
o888ooooood8 o88    888o o888o         Y8bod8P  d888b       888
```

Where port is a synonym for gate

ExPort is a performance focused derivative of Expert Gate, a dynamic architecture to eliminate catastrophic forgetting.

<!-- toc -->

- [Features](#features)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
    - [Setup](#setup)
    - [First run](#first-run)
    - [Settings](#settings)
- [Paper](#paper)
- [The Team](#the-team)
- [Credits](#credits)
- [Licence](#licence)

<!-- tocstop -->

## Features
Multiple modes to run the dataset
   - train - trains a model
   - test  - tests the model
   - run   - both trains and tests the model

## Requirements
   - Python 3.7.9 (other versions might work but have not been tested)
   - Preferabily one or more Nvidia GPUs


## Getting Started

### Setup
To get started we need to install some dependencies. This can be done with the following command.

```
pip install pytorch gdown tqdm
```


### First run
```
python Main.py
```

optionally you can use 

```
python Main.py --mode [mode]
```

where [mode] is one of the modes listed above


### Settings
If you do not want the nations dataset or if you are offline, the download can be disabled with the following option:
```python
parser.add_argument("--download_dataset",	type=str,	default="False",	help="Whether to (re-)download dataset")
```

## Paper
The paper can be found [here](https://www.overleaf.com/project/63fc8b9ca583b9a9ad6e04fc) !!change link when its done!!

## The Team
We are a group of 8th semester Computer Science students from Aalborg University, who were tasked at creating "innovative and scalable systems".

## Credits
Our thanks goes out to these libraries and datasets
[Expert Gate](https://github.com/rahafaljundi/Expert-Gate) and their [paper](https://arxiv.org/pdf/1611.06194.pdf) for the inspiration of the project

[Pytorch implementation of Expert Gate](https://github.com/wannabeOG/ExpertNet-Pytorch)

[Pytorch](https://github.com/pytorch/pytorch) and their [paper](https://arxiv.org/abs/1912.01703) for providing a general framework for machine learning

[Mnist](https://drive.google.com/uc?id=1F6xICWB2ZqUouf274xJViMmMvLKC39fA) Mnist Dataset

[Tiny imagenet](http://cs231n.stanford.edu/tiny-imagenet-200.zip) tiny imagenet



## Licence
Copyright 2023 Anders Martin Hansen, Frederik Stær, Frederik Marinus Trudslev, Silas Oliver Torup Bachmann

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
