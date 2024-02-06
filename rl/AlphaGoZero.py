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
                 base_learning_rate=3e-4,
                 max_learning_rate=3e-4,
                 step_size_up=1000,
                 step_size_down=1000,
                 l2_weight=0,
                 transformer_head_dim=64,
                 transition_buffer_len=1000,
                 epsilon=0.9,
                 epsilon_max=0.9,
                 epsilon_delta_per_step=0.00001,
                 num_bins=10,
                 num_acting_player_fields=9,
                 num_other_player_fields=3,
                 save_train_data=False,
                 default_train_file_path='data/test.txt',
                 num_inference_per_step=200,
                 num_data_print_per_inference=500
                 ):
        super(AlphaGoZero, self).__init__()

        self.n_observation = n_observation
        self.n_actions = n_actions
        self.num_output_class = num_output_class
        self.device = torch.device(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_delta_per_step = epsilon_delta_per_step
        self.save_train_data = save_train_data
        self.default_train_file_path = default_train_file_path
        self.num_inference_per_step = num_inference_per_step
        self.num_data_print_per_inference = num_data_print_per_inference

        self.train_file = None

        self.epsilon = epsilon
        self.epsilon_max = epsilon_max
        self.random_choice = np.arange(0, self.n_actions)

        self.model = TransformerAlphaGoZeroModel(num_bins, num_output_class, embedding_dim, positional_embedding_dim, num_layers, transformer_head_dim, historical_action_sequence_length, num_acting_player_fields, num_other_player_fields, device).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=max_learning_rate, weight_decay=l2_weight)
        # self.optimizer = torch.optim.RMSprop(params=[{'params': self.model.parameters()}], lr=learning_rate, weight_decay=l2_weight)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=base_learning_rate, max_lr=max_learning_rate, step_size_up=step_size_up, step_size_down=step_size_down, mode="triangular", cycle_momentum=False)

        self.action_prob_loss = torch.nn.CrossEntropyLoss()
        self.reward_value_loss = torch.nn.MSELoss()
        self.winning_prob_loss = torch.nn.MSELoss()

        self.memory_lock = Lock()
        self.memory = SimpleMemory(transition_buffer_len, n_observation * 2 + 2)

        if self.save_train_data:
            self.train_file = open(default_train_file_path, 'a', encoding='UTF-8')

        self.train_step_num = 0

    def __del__(self):
        if self.save_train_data and self.train_file is not None:
            self.train_file.flush()
            self.train_file.close()

    def forward(self, x):
        return self.model(x)

    def store_transition(self, observation, action_probs, action_masks, reward_value, winning_prob, save_train_data=True):
        if self.save_train_data and save_train_data:
            observation_str = ','.join([str(i) for i in observation.tolist()])
            action_prob_str = ','.join(['%.5f' % prob for prob in action_probs])
            action_mask_str = ','.join(['%d' % action_mask for action_mask in action_masks])
            reward_value_str = '%.8f' % reward_value
            winning_prob_str = '%.8f' % winning_prob
            result_str = f'{observation_str}\n{action_prob_str}\n{action_mask_str}\n{reward_value_str}\n{winning_prob_str}\n\n'
            self.train_file.write(result_str)

        with self.memory_lock:
            self.memory.store([observation, action_probs, action_masks, reward_value, winning_prob])

    def learn(self, observation_list=None, action_probs_list=None, action_masks_list=None, reward_value_list=None, winning_prob_list=None):
        if observation_list is None or action_probs_list is None or action_masks_list is None or reward_value_list is None or winning_prob_list is None:
            sample_idx_np, _, data_batch = self.memory.sample(self.batch_size)
            # weight_tensor = torch.from_numpy(weight_np).to(self.device)

            observation_list = list()
            action_probs_list = list()
            action_masks_list = list()
            reward_value_list = list()
            winning_prob_list = list()
            for observation, action_probs, masks, reward_value, winning_prob in data_batch:
                observation_list.append(observation)
                action_probs_list.append(action_probs)
                # action_masks_list.append(masks)
                reward_value_list.append(reward_value)
                winning_prob_list.append(winning_prob)

        observation_array = np.array(observation_list)
        action_probs_array = np.array(action_probs_list)
        # action_masks_array = np.array(action_masks_list)
        reward_value_array = np.array(reward_value_list)
        winning_prob_array = np.array(winning_prob_list)
        observation_tensor = torch.tensor(observation_array, dtype=torch.int32, device=self.device, requires_grad=False)
        action_probs_tensor = torch.tensor(action_probs_array, dtype=torch.float32, device=self.device, requires_grad=False)
        # action_masks_tensor = torch.tensor(action_masks_array, dtype=torch.int32, device=self.device, requires_grad=False).bool()
        reward_value_tensor = torch.tensor(reward_value_array, dtype=torch.float32, device=self.device, requires_grad=False)
        winning_prob_tensor = torch.tensor(winning_prob_array, dtype=torch.float32, device=self.device, requires_grad=False)

        self.model.train()
        action_prob_logits, reward_value_logits, winning_prob_logits = self.model(observation_tensor)

        action_probs_loss = self.action_prob_loss(action_prob_logits, action_probs_tensor)
        reward_value_loss = self.reward_value_loss(reward_value_logits, reward_value_tensor)
        winning_prob_loss = self.winning_prob_loss(winning_prob_logits, winning_prob_tensor)
        action_probs_loss_float = action_probs_loss.item()
        reward_value_loss_float = reward_value_loss.item()
        winning_prob_loss_float = winning_prob_loss.item()
        over_all_loss = action_probs_loss + reward_value_loss + winning_prob_loss
        assert not torch.any(torch.isnan(over_all_loss)), ValueError('loss is nan')

        self.optimizer.zero_grad()
        over_all_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        # self.scheduler.get_last_lr()

        # del over_all_loss, action_probs_loss, reward_value_loss, winning_prob_loss, observation_tensor, action_probs_tensor, reward_value_tensor, winning_prob_tensor, action_prob, reward_value_logits, winning_prob_logits
        # torch.cuda.empty_cache()

        # v_error_abs_tensor = torch.abs(value_eval_tensor - value_target_tensor)
        # abs_error_np = v_error_abs_tensor.detach().cpu().numpy()
        # self.memory.batch_update(sample_idx_np, abs_error_np)

        # if self.epsilon_delta_per_step is not None and self.epsilon_max is not None and self.epsilon_delta_per_step > 0 and 0 <= self.epsilon_max <= 1:
        #     if self.epsilon < self.epsilon_max:
        #         self.epsilon += self.epsilon_delta_per_step

        self.train_step_num += 1
        if self.train_step_num % self.num_inference_per_step == 0:
            predict_action_probs_tensor = torch.softmax(action_prob_logits, dim=1)
            predict_action_probs_list = predict_action_probs_tensor.cpu().detach().numpy()
            predict_reward_value_list = reward_value_logits.cpu().detach().numpy()
            predict_winning_prob_list = winning_prob_logits.cpu().detach().numpy()
            for data_idx, (action_probs, reward_value, winning_prob, predict_action_probs, predict_reward_value, predict_winning_prob, masks) in enumerate(zip(action_probs_list, reward_value_list, winning_prob_list, predict_action_probs_list, predict_reward_value_list, predict_winning_prob_list, action_masks_list)):
                if data_idx >= self.num_data_print_per_inference:
                    break
                logging.info(f'predictions:\naction_probs={",".join(["%.4f" % item for item in action_probs])}\npredict_action_probs={",".join(["%.4f" % item for item in predict_action_probs.tolist()])}\nreward_value={reward_value}\npredict_reward_value={predict_reward_value}\nwinning_prob={winning_prob}\npredict_winning_prob={predict_winning_prob}\n')

        return action_probs_loss_float, reward_value_loss_float, winning_prob_loss_float
