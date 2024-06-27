# Autonomous Driving RL Policy

This repository contains the backend code for an Autonomous Driving RL Policy project, which uses reinforcement learning techniques to simulate and optimize driving strategies for autonomous vehicles.

## Project Overview

This project aims to develop and test RL algorithm (Q-learning) in simulated environments that mimic real-world driving scenarios. The ultimate goal is to improve decision-making processes for autonomous vehicles under various traffic conditions.

## Getting Started

Follow these instructions to get a copy of the project running on your local machine for development and testing purposes.

### Prerequisites

You'll need Python 3.x installed on your machine, along with the following libraries:
- NumPy
- OpenCV for Python
- Matplotlib

You can install these with pip:

```bash
pip install numpy opencv-python matplotlib

## Configuration and Usage

The simulation is configurable through several command-line arguments, allowing adjustments to the environment, learning parameters, and simulation details. Here’s how to use the available options:

- `--size`: Specifies the size of the environment grid. Default is 7x7.
- `--episodes`: Sets the total number of episodes to run during the simulation. Default is 250,000.
- `--show_every`: Determines how often the environment should be visually updated (in terms of episodes). Default is every 3000 episodes.
- `--move_penalty`: Penalty for each move made by a vehicle that does not result in reaching the target. Default is 1.
- `--target_reward`: Reward received for reaching the target. Default is 25.
- `--epsilon_decay`: Rate at which the exploration rate decays. Default is 0.9998.
- `--start_epsilon`: Starting exploration rate, dictating how often random actions are taken. Default is 0.9.
- `--learning_rate`: Learning rate for the Q-learning update rule. Default is 0.1.
- `--discount`: Discount factor used in the Q-learning update, representing the difference in importance between future rewards and immediate rewards. Default is 0.95.
- `--qtable_filename`: Optional filename to load a pre-existing Q-table, allowing for continuation from previous training. Default is `None`.
- `--obstacle_penalty`: Penalty for encountering an obstacle or hazard. Default is 300.
- `--num_dynamic_obstacles`: Number of dynamic obstacles (moving vehicles) in the environment. Default is 3.
- `--num_static_obstacles`: Number of static obstacles in the environment. Default is 2.

To run the simulation with custom settings, you might use a command like this:

```bash
python main.py --size 10 --episodes 50000 --show_every 1000 --move_penalty 2 --target_reward 50 --epsilon_decay 0.9995 --start_epsilon 0.8 --learning_rate 0.05 --discount 0.99 --num_dynamic_obstacles 5 --num_static_obstacles 3