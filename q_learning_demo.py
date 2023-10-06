import random
import os
import numpy as np
import cv2
from PIL import Image
import time
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

EPSILON = 0.65
ALPHA = 0.1
GAMMA = 0.95

EPISODES = 30000
EVERY_SHOW = 3000

FOOD_REWARD = 25
ENEMY_PENALTY = -300
STEP_PENALTY = -1

SIZE = 10
ACT_SPACE_N = 4

color = {
    'background': (0, 0, 0),
    'player': (255, 0, 0),  # blue
    'food': (0, 255, 0),  # green
    'enemy': (0, 0, 255)  # red
}


class Agent(object):
    def __init__(self, size):
        self.size = size
        self.x = np.random.randint(0, size-1)
        self.y = np.random.randint(0, size-1)

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    @staticmethod
    def act(state, q_table):
        if EPSILON > np.random.random(1):
            action = np.argmax(q_table[state])
        else:
            action = np.random.randint(0, ACT_SPACE_N)
        return action

    @staticmethod
    def learn(state, action, reward, next_state, q_table):
        if reward == FOOD_REWARD:
            q_table[state][action] = FOOD_REWARD
        elif reward == -ENEMY_PENALTY:
            q_table[state][action] = -ENEMY_PENALTY
        else:
            predict_q = q_table[state][action]
            target_q = reward + GAMMA * max(q_table[next_state])
            q_table[state][action] += ALPHA * (target_q - predict_q)

    def move(self, action):
        if action == 0:
            self.x += 1
        elif action == 1:
            self.x -= 1
        elif action == 2:
            self.y += 1
        elif action == 3:
            self.y -= 1

        if self.x > self.size - 1:
            self.x = self.size - 1
        elif self.x < 0:
            self.x = 0
        if self.y > self.size - 1:
            self.y = self.size - 1
        elif self.y < 0:
            self.y = 0


class Environment(object):
    def __init__(self):
        self.enemy = None
        self.player = None
        self.food = None
        self.episode_step = 0

    @staticmethod
    def get_table(qtable_name=None):
        if qtable_name is None:
            q_table = {}
            for x1 in range(-SIZE + 1, SIZE):
                for y1 in range(-SIZE + 1, SIZE):
                    for x2 in range(-SIZE + 1, SIZE):
                        for y2 in range(-SIZE + 1, SIZE):  # (5 * np.random.rand(5) - 5)
                            q_table[(x1, y1, x2, y2)] = [random.randint(-5, 0) for i in range(4)]
        else:
            with open(qtable_name, 'rb') as file:
                q_table = pickle.load(file)
        return q_table

    def reset(self):
        self.episode_step = 0
        self.enemy = Agent(SIZE)
        self.player = Agent(SIZE)
        self.food = Agent(SIZE)
        state = (self.player - self.food) + (self.player - self.enemy)
        return state

    def step(self, action):
        self.episode_step += 1
        self.player.move(action)
        next_state = (self.player - self.food) + (self.player - self.enemy)

        if self.player == self.food:
            reward = FOOD_REWARD
        elif self.player == self.enemy:
            reward = ENEMY_PENALTY
        else:
            reward = STEP_PENALTY

        if self.player == self.food or self.player == self.enemy or self.episode_step >= 200:
            done = True
        else:
            done = False
        return next_state, reward, done

    def render(self):
        env_array = np.full((SIZE, SIZE, 3), color['background'], dtype=np.uint8 )
        env_array[env.player.x, env.player.y] = color['player']
        env_array[env.food.x, env.food.y] = color['food']
        env_array[env.enemy.x, env.enemy.y] = color['enemy']
        img = Image.fromarray(env_array, "RGB")
        img = img.resize((500, 500))
        cv2.imshow('Game', np.asarray(img))
        if self.player == self.food or self.player == self.enemy:
            cv2.waitKey(200)
        else:
            cv2.waitKey(1)


def train_episode(env, is_render, q_table):
    state = env.reset()
    episode_reward = 0
    while True:
        action = env.player.act(state, q_table)
        next_state, reward, done = env.step(action)
        env.player.learn(state, action, reward, next_state, q_table)
        state = next_state
        episode_reward += reward
        if is_render:
            env.render()
        if done:
            break
    return episode_reward


def plot_result(x_length, y):
    x = [i for i in range(x_length)]
    plt.plot(x, y)
    plt.xlabel('episode')
    plt.ylabel(f'mean {EVERY_SHOW} rewards')
    plt.title('Training Curve')
    plt.show()


def train(env):
    total_rewards = []
    qtable_name = 'q_table_5-11-40-53.pickle'
    q_table = env.get_table(qtable_name)
    for e in range(EPISODES):
        if e % EVERY_SHOW == 0:
            is_render = True
            print(f' episode = {e}, total_rewards = {np.mean(total_rewards[-EVERY_SHOW:])} ')
        else:
            is_render = False
        episode_reward = train_episode(env, is_render, q_table)
        total_rewards.append(episode_reward)

    moving_avg = np.convolve(total_rewards, np.ones(EVERY_SHOW)/EVERY_SHOW, mode='valid')
    plot_result(len(moving_avg), moving_avg)
    lt = time.localtime()
    record_tm = f'{lt.tm_mday}-{lt.tm_hour}-{lt.tm_min}-{lt.tm_sec}'

    with open(f'q_table_{record_tm}.pickle', 'wb') as file2:
        print(f"已创建新文件: q_table_{record_tm}.pickle")
        pickle.dump(q_table, file2)

    if os.path.exists(qtable_name):
        os.remove(qtable_name)
        print(f"已删除上一个文件: {qtable_name}")


if __name__ == '__main__':
    env = Environment()
    train(env)
