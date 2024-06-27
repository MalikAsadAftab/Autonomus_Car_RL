from config import parse_args
from vehicle import Vehicle, initialize_obstacles
from environment import setup_game
from deepqlearning import create_model, update_dqn
from visualization import show_environment, reward_figure
from utils import get_state, compute_reward
import numpy as np
import os
import time
import csv
import logging

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

def main(args):
    # Configure logging to help with debugging and tracking the simulation progress.
    logging.basicConfig(level=logging.INFO)

    # Setup and extract arguments for the simulation.
    size = args.size
    episodes = args.episodes
    show_every = args.show_every
    move_penalty = args.move_penalty
    obstacle_penalty = args.obstacle_penalty  
    target_reward = args.target_reward
    epsilon_decay = args.epsilon_decay
    start_epsilon = args.start_epsilon
    learning_rate = args.learning_rate
    discount = args.discount

    # Number of dynamic and static obstacles are hardcoded, consider adding these to argparse if needed for flexibility.
    num_dynamic_obstacles = args.num_dynamic_obstacles
    num_static_obstacles = args.num_static_obstacles

    # Directory setup for saving Q-tables, episodes outputs, and reward plots.
    qtable_directory = 'qtables'
    os.makedirs(qtable_directory, exist_ok=True)
    qtable_path = os.path.join(qtable_directory, 'qtable-{}.pickle'.format(int(time.time())))

    episode_run_directory = 'episode_run_output'
    os.makedirs(episode_run_directory, exist_ok=True)
    csv_file_path = os.path.join(episode_run_directory, 'episode_outputs.csv')

    reward_plot = 'reward_plots'
    os.makedirs(reward_plot, exist_ok=True)

    # Initialize or load Q-table based on provided filename.
    #q_table = initialize_q_table(size) if not args.qtable_filename or not os.path.exists(args.qtable_filename) else pickle.load(open(args.qtable_filename, "rb"))

    # Inititalize the Deep Q learning
    state_size = 6  # As defined by get_state function
    action_size = 4  # Assuming four possible actions
    model = create_model(state_size, action_size)

    # Prepare CSV file to log episode results.
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Epsilon', 'Episode Reward', 'Mean Reward'])

        epsilon = start_epsilon
        episode_rewards = []

        # Simulation loop for the given number of episodes.
        for episode in range(episodes):
            player = Vehicle(size, is_dynamic=True)
            target = Vehicle(size, is_dynamic=False)  # Assuming the target is static.
            obstacles = initialize_obstacles(num_dynamic_obstacles, num_static_obstacles, size)

            if episode % show_every == 0:
                logging.info(f"on #{episode}, epsilon is {epsilon}")
                if episode_rewards:
                    logging.info(f"{show_every} ep mean: {np.mean(episode_rewards[-show_every:])}")
                show = True
            else:
                show = False

            episode_reward = 0

            # Define colors for different entities for visualization.
            d = {1: (255, 175, 0), 2: (0, 255, 0), 3: (0, 0, 255)}

            for _ in range(200):
                # Compute the current state
                state = get_state(player, target, obstacles[0])
                action = np.argmax(model.predict(state[np.newaxis])[0]) if np.random.random() > epsilon else np.random.randint(0, 4)
                player.action(action)

                # Process actions for each obstacle, whether static or dynamic.
                for obstacle in obstacles:
                    obstacle_action = np.random.randint(0, 4)
                    obstacle.action(obstacle_action)  # Apply action to each obstacle

                # Compute the new state and reward
                new_state = get_state(player, target, obstacles[0])
                reward = compute_reward(player, target, obstacles[0], move_penalty, obstacle_penalty, target_reward)

                # Update the model
                update_dqn(model, state, action, new_state, reward, learning_rate, discount)

                if show:
                    entities = [(1, player.x, player.y), (2, target.x, target.y)]
                    entities.extend((3, obs.x, obs.y) for obs in obstacles)
                    env = setup_game(size, d, entities)
                    show_environment(env, entities)

                episode_reward += reward
                if reward == target_reward or reward == -obstacle_penalty:
                    break

            epsilon *= epsilon_decay
            episode_rewards.append(episode_reward)
            mean_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            writer.writerow([episode, epsilon, episode_reward, mean_reward])

        # Visualize reward dynamics over episodes and save the updated Q-table.
        reward_figure(episode_rewards, show_every, reward_plot)

if __name__ == "__main__":
    args = parse_args()
    main(args)
