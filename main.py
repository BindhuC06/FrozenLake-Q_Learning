import numpy as np
import random
import gymnasium as gym
import matplotlib.pyplot as plt

#Initialization 
env=gym.make("FrozenLake-v1",is_slippery=False)

episodes=10000
state_size = env.observation_space.n
action_size = env.action_space.n

q_table = np.zeros((state_size, action_size))
learning_rate=0.9
discount_factor=0.99
epsilon=1
max_ep=1
min_ep=0.01
decay_rate=0.0005

ep_rewards=[]

# Training loop
for ep in range(episodes):
    state,info=env.reset()
    done=False
    total_reward=0

    while not done:
        # 1.Epsilon-greedy action selection
        exp_exp = random.uniform(0, 1)
        if exp_exp>epsilon:
            # Exploit
            action = np.argmax(q_table[state, :])
        else:
            # Explore
            action = env.action_space.sample()
        # 2. Take action, observe new state and reward
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward+=reward
        # 3. Update Q-table using Bellman  Optimality Eqn
        q_table[state, action]=q_table[state, action]+learning_rate*(reward+discount_factor*np.max(q_table[next_state, :])-q_table[state, action])
        state = next_state

    ep_rewards.append(total_reward)
    # 4. Reduce epsilon (less exploration as time goes on)
    epsilon =min_ep+(max_ep-min_ep)*np.exp(-decay_rate * ep)

print("Training finished.\n")
print("Final Q-Table:")
print(np.round(q_table, 4))

window_size = 100
moving_avg_rewards = np.convolve(
    ep_rewards, np.ones(window_size) / window_size, mode="valid"
)

# 5. Plotting the Learning Curve
plt.style.use("seaborn-v0_8-whitegrid")
plt.figure(figsize=(10, 7))
plt.plot(
    range(window_size - 1,episodes),
    moving_avg_rewards,
    color="#1f77b4",
    linewidth=2,
    label="Moving Avg (Window=100)",
)

plt.title("Q-Learning Performance on FrozenLake-v1", fontsize=14, fontweight="bold")
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Win Rate", fontsize=12)
plt.ylim(-0.05, 1.05)
plt.legend(loc="lower right", fontsize=11)
plt.tight_layout()

# Save plot to current working directory
plt.savefig("frozenlake_win_rate_0.1_lr_0.99_discount.png", dpi=300)
print("Performance graph saved.")