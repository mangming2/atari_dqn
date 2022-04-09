import copy
from collections import namedtuple
from itertools import count
import math     # 에이전트가 무작위로 행동할 확률을 구하기 위함
import random   # 에이전트가 무작위로 행동할 확률을 구하기 위함
import numpy as np
import time

import gym      # 게임 환경 패키지 제공

# 파일 이름 (**can be changed)
from wrappers import *
from memory import ReplayMemory
from models import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

Transition = namedtuple('Transion', ('state', 'action', 'reward', 'next_state'))

# epsilon value threshold
def choose_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EP_END + (EP_START - EP_END) * math.exp(-1. * steps_done / EP_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to('cuda')).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)

# batch sample train
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))
    actions = tuple((map(lambda a: torch.tensor([[a]] ,device='cuda'), batch.action)))
    rewards = tuple((map(lambda r: torch.tensor([r], device='cuda'), batch.reward)))

    non_last_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
            device=device, dtype=torch.uint8)

    non_last_next_states = torch.cat([s for s in batch.next_state if s is not None]).to('cuda')

    state_batch = torch.cat(batch.state).to('cuda')
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_last_mask] = target_net(non_last_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
    
# state
def get_state(obs):
    state = np.array(obs)
    # state = state[35:195, :, :]
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)

# train loop
def train(env, n_episodes, render=False):
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        
        # until done  ~ count
        for step in count():
            # select action 할때마다 steps_done += 1
            action = choose_action(state)

            if render:
                env.render()

            obs, reward, done, info = env.step(action)
            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            reward = torch.tensor([reward], device=device)
            
            # push data in replay memory
            # 지선 수정 cpu -> cuda 변경
            memory.push(state, action.to('cuda'), reward.to('cuda'), next_state)
            state = next_state

            # until not train model
            if steps_done > INITIAL_MEMORY:
                optimize_model()
                
                # 1000's step -> target-net update
                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done:
                break
        # 10 episode마다 중간 결과값 출력 (지선 수정 : 50 -> 10)
        if episode % 10 == 0:
            print('total steps: {} \t episodes: {}/{} \t total reward: {}'.format(steps_done, episode, step, total_reward))

    # training 확인 용도
    print('model training is complete!!!\n')
    env.close()
    return

# trained-model -> tset episode
# 지선 수정 render = False 수정
def test(env, n_episodes, policy, render=False):
    # 저장되는 영상 이름
    env = gym.wrappers.Monitor(env, './videos/' + 'dqn_breakoutNo_video')
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        # 지선 수정 t -> step
        for step in count():
            action= policy(state.to('cuda')).max(1)[1].view(1,1)
            
            if render:
                env.render()
                time.sleep(0.02)
            
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            if not done:
                next_state = get_state(obs)
            else:
                next_state = None
                
            state = next_state
            
            if done:
                print("Finished Episode {} with reward {}".format(episode, total_reward))
                break
    env.close()
    return

if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 32             
    GAMMA = 0.99                # 할인 계수 : 에이전트가 현재 reward를 미래 reward보다 얼마나 더 가치있게 여기는지
    EP_START = 1                # 학습 시작 시, 에이전트가 무작위로 행동할 확률
    EP_END = 0.02               # 학습 막바지에 에이전트가 무작위로 행동할 확률

    ''' 랜덤하게 하는 이유는 Agent가 가능한 모든 행동을 경험하기 위함.
        EP_START부터 EP_END까지 점진적으로 감소시켜줌.
        --> 초반에는 경험을 많이 쌓게 하고, 점차 학습하면서, 학습한대로 진행하게끔
    '''
    
    # (지선 수정 : 1000000 -> 200)
    EP_DECAY = 200          # 학습 진행시 에이전트가 무작위로 행동할 확률을 감소시키는 값
    TARGET_UPDATE = 1000
    RENDER = False
    lr = 1e-4                   # 학습률
    INITIAL_MEMORY = 10000
    MEMORY_SIZE = 10 * INITIAL_MEMORY
    
    # game action space -> n_actions
    policy_net = DQN(n_actions=4).to(device)
    target_net = DQN(n_actions=4).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    steps_done = 0

    env = gym.make('BreakoutNoFrameskip-v4')
    env = make_env(env)

    memory = ReplayMemory(MEMORY_SIZE)

    # episode수정
    train(env, 400)
    # 저장되는 모델 이름
    torch.save(policy_net, "dqn_breakoutNo_model")
    # 모델 불러오기
    policy_net = torch.load("dqn_breakoutNo_model")
    test(env, 1, policy_net, render=False)
