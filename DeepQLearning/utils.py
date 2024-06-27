import numpy as np

def get_state(player, target, obstacle):
    # Make sure to include all necessary components of the state
    # This example assumes player, target, and obstacle each have x and y coordinates.
    # Adjust accordingly if your state includes other information.
    return np.array([player.x, player.y, target.x, target.y, obstacle.x, obstacle.y], dtype=float)

def compute_reward(player, target, obstacle, move_penalty, obstacle_penalty, target_reward):
    """
    Calculate the reward for the player based on the current state.

    Args:
    player (Vehicle): The player vehicle object.
    target (Vehicle): The target vehicle object (static or dynamic).
    obstacle (Vehicle): The obstacle vehicle object.
    move_penalty (int): Penalty for moving without reaching the target or hitting an obstacle.
    obstacle_penalty (int): Penalty for colliding with an obstacle.
    target_reward (int): Reward for reaching the target.

    Returns:
    int: The reward based on the player's actions.
    """
    if player.x == obstacle.x and player.y == obstacle.y:
        return -obstacle_penalty
    elif player.x == target.x and player.y == target.y:
        return target_reward
    else:
        return -move_penalty
