
#cartpole-game.py__________Actor-Critic__________CartPole-v0

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib import animation

np.random.seed(2)
torch.manual_seed(2)  # reproducible



# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")





OUTPUT_GRAPH = False
MAX_EPISODE = 500
DISPLAY_REWARD_THRESHOLD = 250  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = False
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic


env = gym.make('CartPole-v1', render_mode='rgb_array')
env.reset(seed=1)  # reproducible

N_F = env.observation_space.shape[0]
N_A = env.action_space.n



class Actor(nn.Module):
    def __init__(self, n_features, n_actions, lr=0.001):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_features, 32)
        self.fc2 = nn.Linear(32, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x

    def learn(self, s, a, td_error):
        self.optimizer.zero_grad()
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
        a = torch.tensor(a, dtype=torch.int64).to(device)
        td_error = torch.tensor(td_error, dtype=torch.float32).to(device)
        probs = self.forward(s)
        log_prob = torch.log(probs.squeeze(0)[a])
        loss = -log_prob * td_error  # negative for gradient ascent
        loss.backward()
        self.optimizer.step()

    def choose_action(self, s):
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
        probs = self.forward(s)
        m = torch.distributions.Categorical(probs)
        return m.sample().item()


class Critic(nn.Module):
    def __init__(self, n_features, lr=0.01):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_features, 32)
        self.fc2 = nn.Linear(32, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def learn(self, s, r, s_):
        self.optimizer.zero_grad()
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
        s_ = torch.tensor(s_, dtype=torch.float32).unsqueeze(0).to(device)
        r = torch.tensor(r, dtype=torch.float32).to(device)
        v = self.forward(s)
        v_ = self.forward(s_).detach()  # one step td
        td_error = r + GAMMA * v_ - v
        loss = td_error.pow(2)
        loss.backward()
        self.optimizer.step()
        return td_error.item()



# Function to create an animation from frames
def create_animation(frames):
    fig = plt.figure()
    plt.axis('off')
    patch = plt.imshow(frames[0])

    def animate(i):
        patch.set_data(frames[i])

    ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=50)
    plt.show()

# 
actor = Actor(n_features=N_F, n_actions=N_A, lr=LR_A).to(device)
critic = Critic(n_features=N_F, lr=LR_C).to(device)

# 
rewards = []
frames=[]
record_episode=0
for i_episode in range(MAX_EPISODE):
    s, _ = env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER and i_episode==record_episode and record_episode!=0: 
            frame=env.render()
            frames.append(frame)
            print(f"{i_episode} frame saved")
        a = actor.choose_action(s)
        s_, r, done, _, _ = env.step(a)
        if done: r = -20
        track_r.append(r)
        td_error = critic.learn(s, r, s_)
        actor.learn(s, a, td_error)
        s = s_
        t += 1
        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD and record_episode==0 and i_episode>150: 
                RENDER = True
                record_episode=i_episode+1
            print("episode:", i_episode, "  reward:", int(running_reward))
            rewards.append(running_reward)
            break



# 显示训练后的小车动画
create_animation(frames)


# 绘制累积奖励随迭代次数的变化曲线
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Running Reward')
plt.title('Running Reward vs Episode')
plt.show()


