## 定义参数 -> 环境变化(目标不变) 状态量是距离差值 | agent变化 获取观测量 ->  Qtable更新
import random
import numpy as np
import cv2
from PIL import Image
import time
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

## 定义参数
SIZE = 10
EPISODES = 30000
EVERY_SHOW = 500
epsilon = 0.6

FOOD_REWARD = 25
ENEMY_PENALTY = 300
STEP_PENALTY = 1

ALPHA = 0.1
GAMMA = 0.95
EPS_DECAY = 0.9998
Q_table = 'Q_table_1696335918.pickle'

colors = {
    'background': (0, 0, 0),
    'player': (255, 0, 0), # blue
    'food': (0, 255, 0), # green
    'enemy': (0, 0, 255) # red
}

class Cuber:
    def __init__(self):
        self.x = np.random.randint(0, SIZE-1)
        self.y = np.random.randint(0, SIZE-1)

    def __sub__(self, other):
        return (self.x-other.x, self.y- other.y)

    def move(self, choose):
        if choose == 0 : # →
            self.x += 1
            self.y += 0
        elif choose == 1 : # ←
            self.x -= 1
            self.y -= 0
        elif choose == 2 : # ↑
            self.x += 0
            self.y += 1
        elif choose == 3 : # ↓
            self.x -= 0
            self.y -= 1

        if self.x < 0 :
            self.x = 0
        elif self.x > SIZE-1 :
            self.x = SIZE-1
        if self.y < 0 :
            self.y = 0
        elif self.y > SIZE-1 :
            self.y = SIZE-1

#定义Qtable
if Q_table is None:
    Q_table={}
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE+1, SIZE):
                for y2 in range(-SIZE+1, SIZE):
                    Q_table[((x1, y1), (x2, y2))] = [random.randint(-5, 0) for i in range(4)]
else:
    with open(Q_table,'rb') as file:
        Q_table = pickle.load(file)

total_rewards = []
for episode in range(EPISODES):
    player = Cuber()
    food = Cuber()
    enemy = Cuber()
    episode_reward = 0

    if episode % EVERY_SHOW == 0:
        show = True
        print(f'episode = {episode}, mean_reward = {np.mean(total_rewards[-EVERY_SHOW:])}')
    else:
        show = False

    for state_num in range(200):
        # ------- state ------- #
        obs = (player - food, player - enemy)
        # ------- action ------- #
        if np.random.random() > epsilon:
            action = np.argmax(Q_table[obs])
            player.move(action)
        else:
            action = np.random.randint(0, 4)
            player.move(action)
        # ------- reward ------- #
        if player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        elif player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        else:
            reward = -STEP_PENALTY
        # ------- state_next ------- #
        obs_next = (player - food, player - enemy)
        # ------- learning ------- #
        if reward == FOOD_REWARD:
            Q_table[obs][action] = FOOD_REWARD
        elif reward == -ENEMY_PENALTY:
            Q_table[obs][action] = -ENEMY_PENALTY
        else:
            Q_max = np.max(Q_table[obs_next])
            Q_table[obs][action] = (1-ALPHA)*Q_table[obs][action] + ALPHA*(reward + GAMMA*Q_max)

        episode_reward += reward

        if show:
            env = np.full((SIZE, SIZE, 3), colors['background'], dtype=np.uint8)
            env[player.x, player.y] = colors['player']
            env[food.x, food.y] = colors['food']
            env[enemy.x, enemy.y] = colors['enemy']
            img = Image.fromarray(env, "RGB")
            img = img.resize((500, 500))
            cv2.imshow('GAME', np.array(img))
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(2) & 0xFF == ord('q'):
                    break

        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break

    epsilon = epsilon * EPS_DECAY
    total_rewards.append(episode_reward)

with open(f'Q_table_{int(time.time())}.pickle','wb') as f:
    pickle.dump(Q_table,f)

moving_avg = np.convolve(total_rewards, np.ones((EVERY_SHOW,)) / EVERY_SHOW, mode='valid')
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.xlabel('episode #')
plt.ylabel(f'mean{EVERY_SHOW} reward')
plt.show()















