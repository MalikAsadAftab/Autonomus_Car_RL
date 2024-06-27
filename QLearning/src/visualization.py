from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time
import logging

logging.basicConfig(level=logging.INFO)

def show_environment(env, entities, title="Game", wait_time=1):
    img = Image.fromarray(env, 'RGB')
    img = img.resize((1200, 1000))  # Consider making size a parameter
    cv2.imshow(title, np.array(img))
    k = cv2.waitKey(wait_time) & 0xFF
    if k == 27:  # Esc key to close the window
        cv2.destroyAllWindows()
        logging.info("Closed the visualization window.")

def reward_figure(episode_rewards, SHOW_EVERY, reward_plot='reward_plots'):
    os.makedirs(reward_plot, exist_ok=True)  # Ensure directory exists

    moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
    plt.figure(figsize=(10, 5))  # Optionally set figure size
    plt.plot([i for i in range(len(moving_avg))], moving_avg, label='Reward Moving Average')
    plt.ylabel(f"Reward {SHOW_EVERY}ma")
    plt.xlabel("Episode #")
    plt.title("Reward Trends Over Episodes")
    plt.grid(True)
    plt.legend()

    # Save the figure
    filename = f"reward_plot_{int(time.time())}.png"  # Save with a unique name
    filepath = os.path.join(reward_plot, filename)
    plt.savefig(filepath)
    print(f"Saved reward plot to {filepath}")

    plt.show()  # Display the plot