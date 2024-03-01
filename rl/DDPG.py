import logging

import torch
import random
import numpy as np
from threading import Lock
from rl.base_rl_model import BaseRLModel
from rl.models import *
from tools.memories.simple_memory import SimpleMemory


class DDPG(nn.Module, BaseRLModel):
    def __init__(self,
                 n_observation,
                 n_actions,
                 device='cpu',
                 gamma=0.9,
                 embedding_dim=512,
                 positional_embedding_dim=128,
                 num_layers=6,
                 historical_action_per_round=56,
                 batch_size=32,
                 actor_learning_rate=1e-3,
                 critic_learning_rate=2e-3,
                 transition_buffer_len=1000,
                 epsilon=0.9,
                 epsilon_max=0.9,
                 epsilon_delta_per_step=0.00001,
                 synchronize_model_per_step=500,
                 num_bins=10,
                 num_player_fields=7,
                 initialize_critic_model=False,
                 save_train_data=False
                 ):
        super(DDPG, self).__init__()

        self.n_observation = n_observation
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_delta_per_step = epsilon_delta_per_step
        self.synchronize_model_per_step = synchronize_model_per_step
        self.initialize_critic_model = initialize_critic_model
        self.save_train_data = save_train_data

        if self.initialize_critic_model:
            self.epsilon = 0.1
            self.epsilon_max = 0.1
            self.random_choice = np.asarray([0, 0, 0, 1, 2, 3])
        else:
            self.epsilon = epsilon
            self.epsilon_max = epsilon_max
            self.random_choice = np.arange(0, self.n_actions)

        self.actor_eval_network = TransformerActorModel(num_bins, embedding_dim, positional_embedding_dim, num_layers, historical_action_per_round, num_player_fields, device).to(self.device)
        self.actor_target_network = TransformerActorModel(num_bins, embedding_dim, positional_embedding_dim, num_layers, historical_action_per_round, num_player_fields, device).to(self.device)
        self.actor_optimizer = torch.optim.Adam(params=[{'params': self.actor_eval_network.parameters()}], lr=actor_learning_rate)

        self.critic_eval_network = TransformerCriticalModel(num_bins, embedding_dim, positional_embedding_dim, num_layers, historical_action_per_round, num_player_fields, device).to(self.device)
        self.critic_target_network = TransformerCriticalModel(num_bins, embedding_dim, positional_embedding_dim, num_layers, historical_action_per_round, num_player_fields, device).to(self.device)
        self.critic_optimizer = torch.optim.Adam(params=[{'params': self.critic_eval_network.parameters()}], lr=critic_learning_rate)
        self.value_critic_loss = torch.nn.MSELoss()
        self.win_rate_critic_loss = torch.nn.MSELoss()

        self.softmax = torch.nn.Softmax(dim=1)

        self.memory_lock = Lock()
        self.memory = SimpleMemory(transition_buffer_len, n_observation * 2 + 2)

        self.current_batch_num = 0

        self.batch_indices = torch.arange(0, self.batch_size, dtype=torch.long)

        if self.save_train_data:
            self.train_file = open('data/test.txt', 'a', encoding='UTF-8')

    def __del__(self):
        if self.save_train_data:
            self.train_file.flush()
            self.train_file.close()

    def choose_action(self, observation):
        if random.random() < self.epsilon:
            observation_list = [observation]
            self.actor_eval_network.eval()
            with torch.no_grad():
                actions, values = self.actor_eval_network(observation_list)
            action = actions.cpu().numpy().tolist()[0]
            value = values.cpu().numpy().tolist()[0]
        else:
            action = np.random.choice(self.random_choice).tolist()
            value = np.random.random()
        return action, value

    def cal_reward(self, observation, action):
        self.critic_eval_network.eval()
        observation_list = [observation]
        action_list = [action[0]]
        action_value_list = [action[1]]
        action_tensor = torch.tensor(action_list, dtype=torch.long).to(self.device)
        action_value_tensor = torch.tensor(action_value_list, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            value_tensor, win_rate_tensor = self.critic_eval_network(observation_list, (action_tensor, action_value_tensor))
        value = value_tensor.cpu().numpy().tolist()[0]
        win_rate = win_rate_tensor.cpu().numpy().tolist()[0]
        return value, win_rate

    def store_transition(self, observation, action, reward, observation_):
        def format_observation(obs):
            cards_ori_list = obs[2]
            cards_list = []
            for cards_ori in cards_ori_list:
                cards_list.extend(cards_ori)
            cards_str = ','.join(str(cards) for cards in cards_list)
            sorted_cards_str = ','.join(str(cards) for cards in obs[3])
            players_str = ','.join(str(player) for player in obs[4])
            return f'{obs[0]};{obs[1]};{cards_str};{sorted_cards_str};{players_str}'

        if self.save_train_data:
            observation_str = format_observation(observation)
            action_str = str(action)
            reward_str = str(reward)
            observation_str_ = format_observation(observation_)
            result_str = f'{observation_str}\n{observation_str_}\n{action_str}\n{reward_str}\n\n'
            self.train_file.write(result_str)
        with self.memory_lock:
            self.memory.store([observation, action, reward, observation_])

    def learn(self):
        torch.backends.cudnn.enabled = False

        if self.current_batch_num % self.synchronize_model_per_step == 0:
            self.actor_target_network.load_state_dict(self.actor_eval_network.state_dict())
            self.critic_target_network.load_state_dict(self.critic_eval_network.state_dict())
            logging.info(f'model synchronized at batch:{self.current_batch_num}')
        self.current_batch_num += 1

        sample_idx_np, _, data_batch = self.memory.sample(self.batch_size)
        # weight_tensor = torch.from_numpy(weight_np).to(self.device)

        observation_list = list()
        action_list = list()
        value_list = list()
        win_rate_list = list()
        reward_list = list()
        observation_list_ = list()
        for observation, action, (win_rate, reward), observation_ in data_batch:
            observation_list.append(observation)
            action_list.append(action[0])
            value_list.append(action[1])
            win_rate_list.append(win_rate)
            reward_list.append(reward)
            observation_list_.append(observation_)
        action_tensor = torch.tensor(action_list, dtype=torch.long).to(self.device)
        value_tensor = torch.tensor(value_list, dtype=torch.float32).to(self.device)
        win_rate_tensor = torch.tensor(win_rate_list, dtype=torch.float32).to(self.device)
        reward_tensor = torch.tensor(reward_list, dtype=torch.float32).to(self.device)

        self.actor_target_network.eval()
        self.critic_target_network.eval()
        action_target_tensor_ = self.actor_target_network(observation_list_)
        value_target_tensor_, _ = self.critic_target_network(observation_list_, action_target_tensor_)

        self.critic_eval_network.train()
        value_eval_tensor, win_rate_eval_tensor = self.critic_eval_network(observation_list, (action_tensor, value_tensor))

        value_target_tensor = reward_tensor + self.gamma * value_target_tensor_
        value_critic_loss = self.value_critic_loss(value_eval_tensor, value_target_tensor)
        win_rate_critic_loss = self.win_rate_critic_loss(win_rate_eval_tensor, win_rate_tensor)
        # critic_loss = value_critic_loss + win_rate_critic_loss
        critic_loss = win_rate_critic_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if not self.initialize_critic_model:
            self.actor_eval_network.train()
            self.critic_eval_network.eval()
            action_tensor = self.actor_eval_network(observation_list)
            v_tensor, _ = self.critic_eval_network(observation_list, action_tensor)
            actor_loss = -torch.mean(v_tensor)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            logging.info(f'actor loss={actor_loss}, value_critic_loss={value_critic_loss}, win_rate_critic_loss loss={win_rate_critic_loss}')
        else:
            logging.info(f'value_critic_loss={value_critic_loss}, win_rate_critic_loss={win_rate_critic_loss}')

        # v_error_abs_tensor = torch.abs(value_eval_tensor - value_target_tensor)
        # abs_error_np = v_error_abs_tensor.detach().cpu().numpy()
        # self.memory.batch_update(sample_idx_np, abs_error_np)

        if self.epsilon_delta_per_step is not None and self.epsilon_max is not None and self.epsilon_delta_per_step > 0 and 0 <= self.epsilon_max <= 1:
            if self.epsilon < self.epsilon_max:
                self.epsilon += self.epsilon_delta_per_step
