import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib import animation

# Set random seed for reproducibility
np.random.seed(2)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define constants
MAX_EP_STEPS = 1000  # maximum time step in one episode
env = gym.make('CartPole-v1', render_mode='rgb_array')
env.reset(seed=1)  # reproducible

# Function to display the CartPole animation
def display_cartpole_random(env, episodes=1):
    frames = []
    for _ in range(episodes):
        state, _ = env.reset()
        for _ in range(MAX_EP_STEPS):
            frame = env.render()
            frames.append(frame)
            action = env.action_space.sample()  # choose a random action
            state, _, done, _, _ = env.step(action)
            if done:
                break
    env.close()
    return frames

# Function to create an animation from frames
def create_animation(frames):
    fig = plt.figure()
    plt.axis('off')
    patch = plt.imshow(frames[0])

    def animate(i):
        patch.set_data(frames[i])

    ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=50)
    plt.show()

# Display and animate the CartPole with random actions
frames = display_cartpole_random(env)
create_animation(frames)


