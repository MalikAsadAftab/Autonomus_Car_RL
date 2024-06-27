import numpy as np
import tensorflow as tf

def create_model(state_size, action_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(state_size,)),  # Adjusted to use Input layer for clarity
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(action_size, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

def update_dqn(model, state, action, new_state, reward, lr, discount):
    current_q_values = model.predict(state[np.newaxis])[0]
    next_q_values = model.predict(new_state[np.newaxis])[0]
    
    # Q-Learning update rule
    current_q_values[action] = (1 - lr) * current_q_values[action] + lr * (reward + discount * np.max(next_q_values))
    
    # Fit the model
    model.fit(state[np.newaxis], current_q_values[np.newaxis], verbose=0)
