Here's the updated README file including the integration of Deep Q-learning:

```markdown
# Autonomous Driving RL Policy

This repository contains the backend code for an Autonomous Driving RL Policy project, which uses reinforcement learning techniques to simulate and optimize driving strategies for autonomous vehicles.

## Project Overview

This project aims to develop and test RL algorithms (Q-learning and Deep Q-learning) in simulated environments that mimic real-world driving scenarios. The ultimate goal is to improve decision-making processes for autonomous vehicles under various traffic conditions.

## Getting Started

Follow these instructions to get a copy of the project running on your local machine for development and testing purposes.

### Prerequisites

You'll need Python 3.x installed on your machine, along with the following libraries:
- NumPy
- OpenCV for Python
- Matplotlib
- TensorFlow

You can install these with pip:

```bash
pip install numpy opencv-python matplotlib tensorflow
```

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
python DeepQLearning/src/main.py --size 10 --episodes 50000 --show_every 1000 --move_penalty 2 --target_reward 50 --epsilon_decay 0.9995 --start_epsilon 0.8 --learning_rate 0.05 --discount 0.99 --num_dynamic_obstacles 5 --num_static_obstacles 3
or 
python QLearning/src/main.py --size 10 --episodes 50000 --show_every 1000 --move_penalty 2 --target_reward 50 --epsilon_decay 0.9995 --start_epsilon 0.8 --learning_rate 0.05 --discount 0.99 --num_dynamic_obstacles 5 --num_static_obstacles 3
```

## Deep Q-learning Integration

This project also integrates Deep Q-learning using TensorFlow. The model is defined in the `deepqlearning.py` module. It uses a neural network to approximate the Q-values, enabling the agent to handle more complex environments with higher-dimensional state spaces.

### Model Architecture

The model consists of a sequential neural network with the following layers:
- Input layer
- Two hidden layers with ReLU activation
- Output layer with a linear activation function

### Example Usage

Here is an example of how the Deep Q-learning model is initialized and used within the main simulation loop:

```python
from deepqlearning import create_model, update_dqn

# Initialize the Deep Q-learning model
state_size = 6  # Modify based on how you define your state space
action_size = 4  # Assuming four possible actions
model = create_model(state_size, action_size)

# Inside the simulation loop
state = get_state(player, target, obstacles[0])
action = np.argmax(model.predict(state[np.newaxis])[0]) if np.random.random() > epsilon else np.random.randint(0, 4)
player.action(action)

# Compute the new state and reward
new_state = get_state(player, target, obstacles[0])
reward = compute_reward(player, target, obstacles[0], move_penalty, obstacle_penalty, target_reward)

# Update the model
update_dqn(model, state, action, new_state, reward, learning_rate, discount)
```

By using Deep Q-learning, the simulation can handle more complex and high-dimensional state spaces, improving the decision-making process for autonomous vehicles.

## Repository Structure

The repository is organized as follows:

```
DeepQLearning
├── config.py
├── deepqlearning.py
├── environment.py
├── main.py
├── utils.py
├── vehicle.py
├── visualization.py
├── qtables/
├── episode_run_output/
├── reward_plots/

QLearning
├── config.py
├── deepqlearning.py
├── environment.py
├── main.py
├── vehicle.py
├── visualization.py
├── qtables/
├── episode_run_output/
├── reward_plots/
```

## Contributions

Contributions are welcome! Please fork the repository and submit pull requests to propose changes or improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
```

You can copy this content into your `README.md` file and push it to your GitHub repository.