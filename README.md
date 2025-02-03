# GameAgents

This repository contains the code for learning agents in games. The agents are implemented in Python and use the Pygame
library for the game environment. The agents are implemented using the reinforcing learn.

## Installation

```shell
conda create -n gameagents python=3.10
```

## Usage

Example usage:

### Train Model

```shell
python main.py --train configs/snake/abstract_state.yaml
```

### Test Model

```shell
python main.py --test configs/snake/abstract_state.yaml
```

![example.png](docs/example.png)



