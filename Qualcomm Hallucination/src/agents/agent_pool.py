import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import math
import pprint

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        self.state_dim = state_dim
        self.device = device
        super(PolicyNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, action_dim),
        ).to(self.device)

    def forward(self, state):
        state = state.float().to(self.device)
        action_logits = self.model(state)
        action_probs = nn.functional.softmax(action_logits, dim=-1)
        # log_probs = torch.log(action_probs)
        return action_probs #, log_probs
    
    def get_log_probs(self, state):
        state = state.float().to(self.device)
        action_logits = self.model(state)
        action_log_probs = nn.functional.log_softmax(action_logits, dim=-1)
        return action_log_probs
    

class PolicyGradientAgent:

    def __init__(self, state_space, action_space, learning_rate, device, map_dict, bert_model):
        
        self.device = device
        self._name = "PolicyGradientAgent"
        self._actions = list(map_dict.values())
        self._policy_network = PolicyNetwork(state_space, action_space, self.device)
        self._optimizer = optim.Adam(self._policy_network.parameters(), lr = learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, step_size=100, gamma=0.9)
        self.bert_model = bert_model
        self._episode_histories = []
        self._action_to_index = {a: i for i,a in enumerate(self._actions)}
        self.state = [None]

    def load(self, path):
        self._policy_network.load_state_dict(torch.load(path))

    def evaluate(self, state):
        self._policy_network.eval()
        with torch.no_grad():
            probs = self._policy_network(state)
        return probs
    
    def change_state(self, current_sentences, dones):
        for idx, (sentence, done) in enumerate(zip(current_sentences, dones)):
            if not done:
                self.states[idx] = self.bert_model.create_embeddings(sentence)

    def get_average_entropies(self, dones, log_action_distributions):
        epsilon = 1e-8
        valid_action_distributions = torch.stack([log_action_distributions[i].exp() for i, done in enumerate(dones) if not done])
        entropies = -torch.sum(valid_action_distributions * torch.log(valid_action_distributions+epsilon), dim=1)
        avg_entropies = torch.mean(entropies)
        return avg_entropies.item()
    
    def choose_action(self, rl_state):

        log_prob = self._policy_network.get_log_probs(rl_state)
        action_dist = dist.Categorical(log_prob.exp())
        action = action_dist.sample()
        chosen_action = self._actions[action.item()]

        return chosen_action, log_prob
    
    def update(self):

        returns = [elem[3] for elem in self._episode_histories]
        returns = torch.tensor(returns)

        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8).clamp(min=0.01)

        else:
            returns = returns * 0

        policy_gradient = []

        for (_, action, action_log_probs, _), ret in zip(self._episodes_histories, returns):

            action_index = self._actions.index(action)
            log_prob_action = action_log_probs[action_index]
            policy_gradient_step = -ret * log_prob_action
            policy_gradient.append(policy_gradient_step)

        policy_gradient = torch.stack(policy_gradient)

        self._optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(self._policy_network.parameters(), 1.0)
        policy_gradient.sum().backward()
        self._optimizer.step()

        self._episode_histories = []


    def discount_rewards(self, gamma):

        all_discounted_rewards = []
        for episode_history in self._episode_histories:

            discounted_rewards = []
            running_add = 0
            for (state, action, action_distribution, reward) in reversed(episode_history):

                if reward != 0:
                    running_add = 0
                running_add = running_add * gamma + reward
                discounted_rewards.insert(0, (state, action, action_distribution, running_add))
            all_discounted_rewards.append(discounted_rewards)

        return all_discounted_rewards
    

    def add_final_rewards(self, rewards):
        for idx, history in enumerate(self._episode_histories):
            last_tuple = history[-1]
            updated_tuple = last_tuple[:-1] + (rewards[idx],)
            history[-1] = updated_tuple

    def step(self, state, action, action_distribution, reward):

        self._episode_histories.append((state, action, action_distribution, reward))


class RandomAgent:
    def __init__(self, num_parallel_envs):

        self._name = "RandomAgent"
        self._actions = [0.2, 0.6, 1.0]
        self._action_to_index = {a: i for i,a in enumerate(self._actions)}
        self._num_parallel_envs = num_parallel_envs
        self._entropy = -math.log2(1.0 / len(self.actions))

    def choose_actions(self, dones):

        chosen_actions = []
        all_probs = []

        for _ in range(self._num_parallel_envs):
            random_action = random.choice(self._actions)
            chosen_actions.append(random_action)

            action_probs = torch.full((1,1,len(self._actions)), 1.0 / len(self._actions))
            all_probs.append(action_probs)

    def get_action_entropy(self):

        eps = 1e-10
        probs = torch.Tensor([[1,0]])
        probs += eps
        entropy = -torch.sum(probs * torch.log2(probs))
        return entropy.item()
    
    def get_average_entropies(self, dones):
        return self._entropy
    
    def change_state(self, current_sentences, dones):
        pass

    def update(self, reward):
        pass

    def step(self, state, action):
        pass
