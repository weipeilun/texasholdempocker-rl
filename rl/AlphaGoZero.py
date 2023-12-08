import logging

import torch
import random
import numpy as np
from threading import Lock
from rl.base_rl_model import BaseRLModel
from rl.models import *
from tools.memories.simple_memory import SimpleMemory


class AlphaGoZero(nn.Module, BaseRLModel):
    def __init__(self,
                 n_observation,
                 n_actions,
                 num_output_class,
                 device='cpu',
                 gamma=0.9,
                 embedding_dim=512,
                 positional_embedding_dim=128,
                 num_layers=6,
                 historical_action_sequence_length=56,
                 batch_size=32,
                 learning_rate=3e-4,
                 l2_weight=0,
                 transformer_head_dim=64,
                 transition_buffer_len=1000,
                 epsilon=0.9,
                 epsilon_max=0.9,
                 epsilon_delta_per_step=0.00001,
                 num_bins=10,
                 num_acting_player_fields=9,
                 num_other_player_fields=3,
                 save_train_data=False
                 ):
        super(AlphaGoZero, self).__init__()

        self.n_observation = n_observation
        self.n_actions = n_actions
        self.num_output_class = num_output_class
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_delta_per_step = epsilon_delta_per_step
        self.save_train_data = save_train_data

        self.epsilon = epsilon
        self.epsilon_max = epsilon_max
        self.random_choice = np.arange(0, self.n_actions)

        self.model = TransformerAlphaGoZeroModel(num_bins, num_output_class, embedding_dim, positional_embedding_dim, num_layers, transformer_head_dim, historical_action_sequence_length, num_acting_player_fields, num_other_player_fields, device).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=l2_weight)
        # self.optimizer = torch.optim.RMSprop(params=[{'params': self.model.parameters()}], lr=learning_rate, weight_decay=l2_weight)

        self.action_prob_loss = torch.nn.CrossEntropyLoss()
        self.action_Q_loss = torch.nn.MSELoss()

        self.memory_lock = Lock()
        self.memory = SimpleMemory(transition_buffer_len, n_observation * 2 + 2)

        if self.save_train_data:
            self.train_file = open('data/test.txt', 'a', encoding='UTF-8')

    def __del__(self):
        if self.save_train_data:
            self.train_file.flush()
            self.train_file.close()

    def forward(self, x):
        return self.model(x)

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

    def cal_reward(self, observation):
        self.model.eval()
        with torch.no_grad():
            action_prob, action_Q_logits = self.model(observation)
        action_prob = action_prob.cpu().numpy().tolist()[0]
        action_Q_logits = action_Q_logits.cpu().numpy().tolist()[0]
        return action_prob, action_Q_logits

    def store_transition(self, observation, action_probs, action_Qs, winning_prob, save_train_data=True):
        if self.save_train_data and save_train_data:
            observation_str = ','.join([str(i) for i in observation.tolist()])
            action_prob_str = ','.join(['%.5f' % prob for prob in action_probs])
            action_Q_str = ','.join(['%.5f' % Q for Q in action_Qs])
            winning_prob_str = '%.8f' % winning_prob
            result_str = f'{observation_str}\n{action_prob_str}\n{action_Q_str}\n{winning_prob_str}\n\n'
            self.train_file.write(result_str)

        with self.memory_lock:
            self.memory.store([observation, action_probs, action_Qs, winning_prob])

    def learn(self, observation_list=None, action_probs_list=None, action_Q_list=None):
        if observation_list is None or action_probs_list is None or action_Q_list is None:
            sample_idx_np, _, data_batch = self.memory.sample(self.batch_size)
            # weight_tensor = torch.from_numpy(weight_np).to(self.device)

            observation_list = list()
            action_probs_list = list()
            action_Q_list = list()
            for observation, action_probs, action_Qs, winning_prob in data_batch:
                observation_list.append(observation)
                action_probs_list.append(action_probs)
                action_Q_list.append(action_Qs)
                # discard winning_prob for model training
                # todo: delete value calcalation process

        action_probs_array = np.array(action_probs_list)
        action_Q_array = np.array(action_Q_list)
        action_probs_tensor = torch.tensor(action_probs_array, dtype=torch.float32, device=self.device, requires_grad=False)
        action_Q_tensor = torch.tensor(action_Q_array, dtype=torch.float32, device=self.device, requires_grad=False)

        self.model.train()
        action_prob, action_Q_logits = self.model(observation_list)

        action_probs_loss = self.action_prob_loss(action_prob, action_probs_tensor)
        action_Q_loss = self.action_Q_loss(action_Q_logits, action_Q_tensor)
        over_all_loss = action_probs_loss + action_Q_loss

        self.optimizer.zero_grad()
        over_all_loss.backward()
        self.optimizer.step()

        # v_error_abs_tensor = torch.abs(value_eval_tensor - value_target_tensor)
        # abs_error_np = v_error_abs_tensor.detach().cpu().numpy()
        # self.memory.batch_update(sample_idx_np, abs_error_np)

        if self.epsilon_delta_per_step is not None and self.epsilon_max is not None and self.epsilon_delta_per_step > 0 and 0 <= self.epsilon_max <= 1:
            if self.epsilon < self.epsilon_max:
                self.epsilon += self.epsilon_delta_per_step

        return action_probs_loss, action_Q_loss
