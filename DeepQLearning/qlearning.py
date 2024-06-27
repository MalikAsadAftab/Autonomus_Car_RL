import numpy as np
import pickle
import os
import logging

def initialize_q_table(size, num_actions=4, default_value=None, filename=None):
    """
    Initialize or load a Q-table.

    Args:
    size (int): Size of the environment.
    num_actions (int): Number of possible actions.
    default_value (float or list): Initial values for Q-table entries.
    filename (str): Path to a pre-existing Q-table file.

    Returns:
    dict: The initialized or loaded Q-table.
    """
    if filename and os.path.exists(filename):
        logging.info(f"Loading Q-table from {filename}.")
        with open(filename, "rb") as f:
            q_table = pickle.load(f)
    else:
        logging.info("Initializing new Q-table.")
        if default_value is None:
            default_value = [np.random.uniform(-5, 0) for _ in range(num_actions)]
        q_table = {}
        for x1 in range(-size + 1, size):
            for y1 in range(-size + 1, size):
                for x2 in range(-size + 1, size):
                    for y2 in range(-size + 1, size):
                        q_table[((x1, y1), (x2, y2))] = default_value.copy()
    return q_table

def update_q_table(q_table, obs, action, new_obs, reward, lr, discount):
    """
    Update a Q-table based on the observation, action taken, and the reward received.

    Args:
    q_table (dict): The Q-table to update.
    obs (tuple): The current state observation.
    action (int): The action taken.
    new_obs (tuple): The new state observation after taking the action.
    reward (float): The reward received for taking the action.
    lr (float): Learning rate.
    discount (float): Discount factor for future rewards.
    """
    max_future_q = max(q_table[new_obs])
    current_q = q_table[obs][action]
    new_q = (1 - lr) * current_q + lr * (reward + discount * max_future_q)
    q_table[obs][action] = new_q
    logging.debug(f"Updated Q-table at {obs}, action {action}: {new_q}")

# Setup basic logging
logging.basicConfig(level=logging.INFO)
