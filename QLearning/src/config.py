import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run Q-learning simulation.")
    parser.add_argument("--size", type=int, default=7, help="Size of the environment (grid size).")
    parser.add_argument("--episodes", type=int, default=250000, help="Number of episodes to run.")
    parser.add_argument("--show_every", type=int, default=3000, help="How often to visualize the environment.")
    parser.add_argument("--move_penalty", type=int, default=1, help="Penalty for moving.")
    parser.add_argument("--target_reward", type=int, default=25, help="Reward for finding target.")
    parser.add_argument("--epsilon_decay", type=float, default=0.9998, help="Rate at which epsilon decays.")
    parser.add_argument("--start_epsilon", type=float, default=0.9, help="Starting value of epsilon.")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate for Q-learning update.")
    parser.add_argument("--discount", type=float, default=0.95, help="Discount factor for Q-learning update.")
    parser.add_argument("--qtable_filename", type=str, default=None, help="Filename from which to load a pre-existing Q-table.")
    parser.add_argument("--obstacle_penalty", type=int, default=300, help="Penalty for encountering a hazard or obstacle.")
    parser.add_argument("--num_dynamic_obstacles", type=int, default=3, help="Number of moving cars")
    parser.add_argument("--num_static_obstacles", type=int, default=2, help="Number of static obstacles")
    return parser.parse_args()
