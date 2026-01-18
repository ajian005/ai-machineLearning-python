"""
经典问题：悬崖寻路
    为了将理论付诸实践，我们将使用一个经典的强化学习示例环境：CliffWalking-v0（悬崖寻路）。它来自 gymnasium 库（原 OpenAI Gym 的维护分支）。

环境描述
    场景：一个 4x12 的网格世界。
    起点：左下角（坐标 [3, 0]）。
    终点：右下角（坐标 [3, 11]）。
    悬崖：最底部一排除了起点和终点的所有位置（[3, 1] 到 [3, 10]），掉入悬崖会获得巨大惩罚并回到起点。
    目标：智能体要从起点安全地走到终点，并避免掉下悬崖。
    动作：上（0）、右（1）、下（2）、左（3）。
    奖励：
        每走一步普通网格：-1（鼓励用更少步数到达）
        掉下悬崖：-100，并被送回起点
        到达终点：0，并结束本次尝试

"""

import gymnasium as gym
import numpy as np
import random

# =========================
# 1. 创建环境
# =========================
env = gym.make("CliffWalking-v1", render_mode="human")

# =========================
# 2. 获取环境信息
# =========================
n_states = env.observation_space.n
n_actions = env.action_space.n

# =========================
# 3. 初始化 Q 表
# =========================
Q_table = np.zeros((n_states, n_actions))

print(f"环境状态数: {n_states}")
print(f"动作数: {n_actions}")
print(f"Q表形状: {Q_table.shape}")

# =========================
# 4. 设置超参数
# =========================
alpha = 0.1
gamma = 0.99
epsilon = 0.1
num_episodes = 500

# =========================
# 5. ε-greedy 策略
# =========================
def choose_action(state, Q_table, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    return np.argmax(Q_table[state])

# =========================
# 6. 训练循环
# =========================
reward_history = []

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = choose_action(state, Q_table, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)

        current_q = Q_table[state, action]

        if terminated:
            target_q = reward
        else:
            target_q = reward + gamma * np.max(Q_table[next_state])

        Q_table[state, action] = current_q + alpha * (target_q - current_q)

        state = next_state
        total_reward += reward

    reward_history.append(total_reward)

    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(reward_history[-100:])
        print(f"轮次 {episode + 1}，最近100轮平均奖励: {avg_reward:.2f}")

env.close()

# =========================
# 7. 测试训练结果
# =========================
print("\n=== 开始测试 ===")

test_env = gym.make("CliffWalking-v1", render_mode="human")
state, _ = test_env.reset()
terminated = False
truncated = False
step_count = 0
total_reward = 0

while not (terminated or truncated):
    action = choose_action(state, Q_table, epsilon=0)
    state, reward, terminated, truncated, _ = test_env.step(action)
    step_count += 1
    total_reward += reward
    print(f"步骤 {step_count}: 状态 {state}, 动作 {action}, 奖励 {reward}")

print(f"测试完成，总步数: {step_count}，总奖励: {total_reward}")

test_env.close()