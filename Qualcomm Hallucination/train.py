import click
from collections import defaultdict
import numpy as np
import torch
import copy
import os
import pprint
import pickle
import matplotlib.pylab as plt
import logging
import math
from pprint import pformat

from environments import Env1
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from agents import PolicyGradientAgent, RandomAgent
from utils import(
    BERTModel, DatasetFetcher, training_plot
)

logging.baseConfig(filename = 'results/terminal_output.txt', label=logging.DEBUG, format='')

torch.cuda.empty_cache()

class ExperimentRunner:

    def __init__(self, database, num_episodes, agent_lr, top_k, language_model_generation, language_model_evaluation, sentence_cutoff, learning_interval, discount_factor, reward_type, mode):

        self.device = database
        self.num_episodes = num_episodes
        self.agent_lr = agent_lr
        self.top_k = top_k
        self.language_model_generation = language_model_generation
        self.language_model_evaluation = language_model_evaluation
        self.sentence_cutoff = sentence_cutoff
        self.learning_interval = learning_interval
        self.discount_factor = discount_factor
        self.reward_type = reward_type
        self.mode = mode

        self.bert_model = BERTModel()

        self.agent_state_space = 768

        self.env_context_window = 1024
        self.map_dict = {0:0.1, 1:0.5, 2:1, 3:1.5}
        self.agent_action_space = len(self.map_dict)

        self.memory_allocated = []
        self.memory_cached = []

        self.total_rewards = []
        self.total_entropies = []
        self.model_entropies = []

    def run(self):

        try:
            self.run_experiment()
        except InterruptedError:
            print("Interrupt!")
        finally:
            print("Saving CUDA results")
            self._plot_cuda_memory_usage()
            print("Saving RAW data")
            self._save_raw_data()
            print("Saving agent")
            self._save_agent("results/agent.pth")
            print("Plotting results")
            self._generate_plots()


    def run_experiment(self):
        self.env = Env1(self.env_context_window, self.top_k, self.language_model_generation, self.language_model_evaluation)
        self._setup_agent()
        self.experiment_loop(self.num_episodes)

    def _track_cuda_memory(self):

        if torch.cuda.is_available():
            self.memory_allocated.append(torch.cuda.memory_allocated() / (1024* 1024))
            self.memory_cached.append(torch.cuda.memory_reserved() / (1024 * 1024))

    def _plot_cuda_memory_usage(self):

        plt.plot(self.memory_allocated, label = "Memory allocated in MB")
        plt.plot(self.memory_cached, label = "Memory Cached in MB")
        plt.xlabel('Steps')
        plt.ylabel('Memory in MB')
        plt.title('CUDA Memory Usage')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/CUDA_memory.png')

    def _save_raw_data(self):
        with open('results.experiment_rewards.pkl', 'wb') as f:
            pickle.dump(self.total_rewards, f)
        with open('results/experiment_entropies.pkl', 'wb') as f:
            pickle.dump(self.total_entropies, f)

    def _setup_agent(self):

        if self.mode == 'train':

            print("Setting up new agent")
            self.agent = PolicyGradientAgent(
                self.agent_state_space, self.agent_action_space,
                learning_rate=self.agent_lr, device = self.device,
                map_dict = self.map_dict, bert_model = self.bert_model
            )
        elif self.mode == 'train':
            print('Loading Agent')
            self._load_agent("src/agents/agent.pth")

    def _save_agent(self, path):
        torch.save(self.agent._policy_network.state_dict(), path)

    def _load_agent(self, path):

        self.agent = PolicyGradientAgent(
            self.agent_state_space, self.agent_action_space,
            learning_rate=self.agent_lr, device = self.device,
            map_dict = self.map_dict, bert_model = self.bert_model
        )
        self.agent._policy_network.load_state_dict(torch.load(path, map_location = self.device))
        self.agent._policy_network.eval()

    def _generate_plots(self):

        training_plot(
            self.average_dataset_diff_reward, "Average differential reward per episode",
            "results/average_diff_rewards.png",
            self.num_episodes, self.agent_lr, self.agent_action_space, self.top_k, self.learning_interval, self.sentence_cutoff, self.discount_factor
        )

        training_plot(
            self.average_dataset_entropy, "Average differential reward per episode",
            "results/average_diff_rewards.png",
            self.num_episodes, self.agent_lr, self.agent_action_space, self.top_k, self.learning_interval, self.sentence_cutoff, self.discount_factor
        )

        training_plot(
            self.average_dataset_model_entropy, "Average differential reward per episode",
            "results/average_diff_rewards.png",
            self.num_episodes, self.agent_lr, self.agent_action_space, self.top_k, self.learning_interval, self.sentence_cutoff, self.discount_factor
        )

        training_plot(
            self.average_dataset_semantic_similarity, "Average differential reward per episode",
            "results/average_diff_rewards.png",
            self.num_episodes, self.agent_lr, self.agent_action_space, self.top_k, self.learning_interval, self.sentence_cutoff, self.discount_factor
        )

    def experiment_loop(self, num_episodes):

        training_size = len(self.database._training_pool)
        shuffled_indices = np.rangom.permutation(training_size)

        self.average_dataset_diff_reward = []
        self.average_dataset_semantic_similarity = []
        self.agerage_dataset_entropy = []
        self.average_dataset_model_entropy = []

        for iteration in range(num_episodes):

            torch.cuda.empty_cache()

            print(f"Iteration {iteration}/{num_episodes}")
            
            sentence_diff_reward = []
            sentence_reward = []
            sentence_entropy = []
            sentence_model_entropy = []

            sent_idx = 0
            for sentence_index in shuffled_indices:
                sent_idx +=1

                print(f"Sentence {sentence_index}/{training_size}")

                initial_prompt = self.database._training_pool.iloc[sentence_index]['instruction']
                golden_output = self.databse._training_pool.iloc[sentence_index]['response']
                category = self.database._training_pool.iloc[sentence_index]['category']

                print("Initial prompt ", initial_prompt)

                avg_episode_reward = 0
                avg_episode_entropy = 0
                avg_episode_model_entropy = 0

                episode_actions = []
                episode_distributions = []

                prev_reward = 0
                max_steps = 0
                done = False
                self.trajectory_length = len(self.env._generation_tokenizer.encode(golden_output, return_tensor="pt")[0])

                self.env.reset(initial_prompt)

                while max_steps <= min(self.trajectory_length, self.sentence_cutoff) and not done:

                    with torch.no_grad():

                        output = self.env_generation_model(
                            input_ids = self.env.input_conditioned,
                            return_dict = True,
                            output_hidden_states = True
                        )

                    gamma_state = 0.9
                    gamma_power = 1
                    gamma_list = []

                    for _ in range(len(output.hidden_states)):

                        gamma_list.append(gamma_power)
                        gamma_power *= gamma_state

                    agent_state = torch.zeros_like(output.hidden_states[len(output.hidden_states)//2])[0][0]
                    gamma_idx = 0
                    for hidden_state in reversed(output.hidden_states):
                        if gamma_idx == 0:
                            agent_state_temp = torch.mean(hidden_state, dim=0)
                        else:
                            agent_state_temp = torch.mean(hidden_state[0], dim=0)
                        agent_state += (gamma_list[gamma_idx]/sum(gamma_list)) * agent_state_temp
                        gamma_idx += 1

                    
                    action, action_log_prob = self.agent.choose_action(agent_state)
                    episode_actions.append(action)
                    episode_distributions.append(action_log_prob)

                    epsilon = 1e-8
                    avg_episode_entropy += -torch.sum(action_log_prob.exp() * torch.log(action_log_prob.exp()+epsilon)).item()

                    logits = output.logits[:,-1]

                    next_reward = 0
                    top_k_values, top_k_indices = torch.topk(logits, self.top_k)

                    model_distribution = Categorical(logits = top_k_values)
                    epsilon = 1e-8
                    model_entropy = -torch.sum(model_distribution.probs * torch.log2(model_distribution.probs + epsilon), dim=-1)

                    avg_episode_model_entropy += model_entropy.item()

                    altered_top_k_values = top_k_values/action
                    distribution = Categorical(logits = altered_top_k_values)
                    sampled_index = distribution.sample()
                    next_token_id = top_k_indices[0][sampled_index].unsqueeze(0)

                    value_tensor = torch.tensor([[next_token_id]])

                    self.env.input_conditioned = torch.cat((self.env.input_conditioned, value_tensor.to(self.device)), dim=-1)
                    self.env.output_generated = torch.cat((self.env.output_generated, value_tensor.to(self.device)), dim=-1)

                    if next_token_id.item() == self.env._generation_tokenizer.eos_token_id:
                        done = True
                    else:
                        done = False

                    next_reward = self.bert_model.check_similarity(golden_output, self.env._generation_tokenizer.decode(self.env.output_generated))

                    diff_reward = next_reward - prev_reward
                    prev_reward = next_reward

                    avg_episode_reward += diff_reward

                    if done or max_steps >= min(self.trajectory_length, self.sentence_cutoff):

                        self.agent.step(agent_state, action, action_log_prob, diff_reward)

                        output_text = self.env._generation_tokenizer.decode(self.env.output_generated)

                        print("Sentences Created: ", output_text)
                        print("Golden output: ", golden_output)

                        reward = self.bert_model.check_similarity(golden_output, output_text)

                        print(f"Took actions: {episode_actions}")
                        print(f"Corresponding distributions: {episode_distributions}")
                        print(f"Produced the sentence: {output_text}")
                        print(f"Semantic Similarity: {reward}")
                        print(f"Average differential reward recieved: {avg_episode_reward/(max_steps+1)}")
                        print(f"Average Model entropy: {avg_episode_model_entropy/(max_steps+1)}")
                        print(f"Average Episode entropy: {avg_episode_entropy/(max_steps+1)}")

                        self._track_cuda_memory()

                        if sent_idx % self.learning_interval == 0:

                            print(f"Learning with {len(self.agent._episode_histories)} samples")
                            self.agent.update()
                        
                        break

                    self.agent.step(agent_state, action, action_log_prob, diff_reward)

                    max_steps +=1
                    self._track_cuda_memory()

                sentence_diff_reward.append(avg_episode_reward/(max_steps+1))
                sentence_reward.append(reward)
                sentence_model_entropy.append(avg_episode_model_entropy/(max_steps+1))
                sentence_entropy.append(avg_episode_entropy/(max_steps+1))

                if avg_episode_model_entropy/(max_steps+1) < 0.000000001:
                    return
                
            
            self.average_dataset_diff_reward.append(sum(sentence_diff_reward)/training_size)
            self.average_dataset_semantic_similarity.append(sum(sentence_reward)/training_size)
            self.agerage_dataset_entropy.append(sum(sentence_entropy)/training_size)
            self.average_dataset_model_entropy.append(sum(sentence_model_entropy)/training_size)

        return
    
    def postprocess_and_save_html_output(self, words_trackers, actions_trackers, parallelization, initial_prompts, categories, avg_rewards):
        pass

@click.command()

@click.option('--num_episodes', default = 300, help='Number of episodes to run')
@click.option('--agent_lr', default = 0.00001, help='Learning rate of policy gradient agent')
@click.option('--learning_interval', default = 10, help='How many episodes before agent learns')
@click.option('--top_k', default = 10, help='How many tokens to consider during token generation')
@click.option('--sentence_cutoff', default = 50, help='Max level of tokens allowed to be generated')
@click.option('--discount_factor', default = 1, help='Discount factor or RL agent')
@click.option('--reward_type', default = 'discrete', help='Type of reward', type=click.Choice(['discrete', 'continuous'], case_sensitive=False))

@click.option('--train_size', default=10, help='Pool of data from Dolly-15k')
@click.option('--test_size', default=10, help='Pool of data from Dolly-15k')

@click.option('--language_model_generation', default='gpt2', help='Choose your model')
@click.option('--language_model_evaluation', default = 'gpt2', help='Evaluation model')

@click.option('--mode', default='train', help='Mode of operation')

def main(num_episodes, agent_lr, learning_interval, top_k, language_model_generation, language_model_evaluation, sentence_cutoff, train_size, test_size, discount_factor, reward_type, mode):

    database_path = 'data/dolly/databricks-dolly-15k.jsonl'
    database = DatasetFetcher(database_path=database_path, train_size = train_size, test_size = test_size)

    if mode == 'test':
        num_episodes = 1

    runner = ExperimentRunner(database, num_episodes, agent_lr, top_k, language_model_generation, language_model_evaluation, sentence_cutoff, learning_interval, discount_factor, reward_type, mode)
    runner.run()


if __name__ == '__main__':
    main()

