# NoisyNet Exploration in QMIX

The primary goal of this repo is to implement the NoisyNet for exploration paper, previously only used with single agent, non-recurrent algorithms, in a multi-agent setting with recurrent neural networks. The secondary goal is to perform benchmark experiments for the algorithm developed during the implementation of [this repo](#https://github.com/dewet99/3d-virtual-environment-qmix). We perform benchmark experiments using the SMACv2 environment and configs.

## Table of Contents
- [Overview](#overview)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Features](#features)
- [To Do](#to-do)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview
We used the original implementation of QMIX as basis for our algorithm, from the [pymarl repo](https://github.com/oxwhirl/pymarl). The SMACv2 benchmark results used a different implementation, with better performance. We therefore benchmark our base algorithm with the same configs used by SMACv2, and compare our benchmark with theirs, and then also compare the performance of our improved algorithm with the SMACv2 benchmark. That is a lot of experiments, and will take quite a while. I'll upload the results as they become available.

## Getting Started
This section will explain how to install everything you need to be able to run experiments using the implemented algorithm and its various configurations.

1. Clone the repo. Later I'll make it into a Docker container for ease-of-use.
    ```bash
    git clone https://github.com/dewet99/combining-improvements-in-marl.git
    ```

2. Install StarCraft II in the combining-improvements-in-marl directory. It will take a while.
    ```bash
    cd combining-improvements-in-marl
    sudo bash install_sc2.sh
    ```

4. Create and activate a python virtual environment.
    ```bash
    python -m venv ./venv
    source ./venv/bin/activate
    ```

4. Install the SMACv2 package
    ```bash
    pip install git+https://github.com/oxwhirl/smacv2.git
    ```

5. Install requirements.txt
    ```bash
    pip install -r requirements.txt
    ```
6. You're done, you should now be able to run experiments.

## Usage
Run the following command in a shell terminal:
```bash
python main.py train -config
```
For config, specify the name of any of the config files in the config directory.
This will generate a `results` directory, which contains tensorboard logs for training. 

## Features
This repo contains the following components for use in the SMACv2 benchmark:
1. Distributed QMIX
2. n-step returns
3. Prioritised experience replay
4. NoisyNet for exploration
5. R2D2's recurrent burn-in and stored hidden state
6. Reward standardisation
We will test each of these components in the benchmark enviroments to verify their contribution towards improving the performance of QMIX.

## To Do and Busy Doing
Run experiments, lots of experiments. In no specific order, I want to check the following:
1. Compare NoisyNet to epsilon-greedy exploration in, at the very least:
    - `protoss_5_vs_5`
    - `protoss_10_vs_10`
    - `protoss_20_vs_20`
    - `protoss_10_vs_11`
    - `protoss_20_vs_23`
2. Perform addition study in at least three environments, namely:
    - `protoss_5_vs_5`
    - `protoss_10_vs_10`
    - `protoss_10_vs_11`
3. Take the best-performing combination of components and train them in the `protoss`, `zerg` and `terran` `10_vs_11`, `20_vs_20` and `20_vs_23` scenarios, to see how my implementation compares to the current state-of-the-art.

## References
- DRQN: [https://arxiv.org/abs/1507.06527](https://arxiv.org/abs/1507.06527)
- R2D2: [https://openreview.net/pdf?id=r1lyTjAqYX](https://openreview.net/pdf?id=r1lyTjAqYX)
- QMIX: [https://arxiv.org/abs/1803.11485](https://arxiv.org/abs/1803.11485)
- RAINBOW: [https://arxiv.org/abs/1710.02298](https://arxiv.org/abs/1710.02298)
- NoisyNet: [https://arxiv.org/abs/1706.10295](https://arxiv.org/abs/1706.10295)
- Prioritised experience replay: [https://arxiv.org/abs/1511.05952](https://arxiv.org/abs/1511.05952)

    
    
        
