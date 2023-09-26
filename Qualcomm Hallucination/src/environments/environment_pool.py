import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.distributions.gategorical import Categorical
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import math

import sys
import time
import pprint

class Env1:

    def __init__(self, context_window, top_k, model_generation, model_evaluation):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_generation == 'gpt2':
            self._generation_model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
            self._generation_model.eval()
            self._generation_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        elif model_generation == 'llama':
            self._generation_model = AutoModelForCausalLM.from_pretrained('llama_path', torch_dtype = torch.float16).to(self.device)
            self._generation_model.eval()
            self._generation_tokenizer = AutoTokenizer.from_pretrained('llama_path')

        if model_evaluation == 'gpt2':
            self._evaluation_model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
            self._evaluation_model.eval()
            self._evaluation_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        elif model_evaluation == 'gpt2':
            self._evaluation_model = GPT2LMHeadModel.from_pretrained('llama_path').to(self.device)
            self._evaluation_model.eval()
            self._evaluation_tokenizer = GPT2Tokenizer.from_pretrained('llama_path')

        self.context_window = context_window
        self.current_words = None
        self.top_k = top_k

    
    def get_metarewards(self, complete_outputs, parallelization, reward_type):

        follow_up_question = 'Is the following answer to the question correct? Yes or no?\n\n'

        full_sentence = [follow_up_question + complete_outputs[i] for i in range(len(complete_outputs))]

        full_tokens = [self._evaluation_tokenizer.encode(sentence, return_tensors = "pt").squeeze().to(self.device) for sentence in full_sentence]

        full_tokens_padded = pad_sequence(full_tokens, batch_first = True)

        original_lengths = [t.size(0) for t in full_tokens]
        attention_mask = full_tokens_padded.ne(0)

        with torch.no_grad():
            outputs = self._evaluation_model(
                input_ids = full_tokens_padded,
                attention_mask = attention_mask,
                return_dict = True
            )

        logits = outputs.logits
        logits = [logits[i, original_lengths[i]-1] for i in range(len(original_lengths))]

        # Yes token is 4874, no token is 694

        ret_rewards = [0] * parallelization
        for idx, logit in enumerate(logits):
            x = logit[4874].item()
            y = logit[694].item()
            max_logit = max(x,y)

            exp_yes = math.exp(x-max_logit)
            exp_no = math.exp(y-max_logit)

            sum_exp = exp_yes + exp_no
            prob_no = exp_no/sum_exp

            ret_rewards[idx] = prob_no

        return ret_rewards
    

    def reset(self, initial_prompt):

        self.input = initial_prompt
        self.input_conditioned = self._generation_tokenizer.encode(initial_prompt, return_tensors = "pt").squeeze(0).to(self.device)
        
        self.output_generated = torch.tensor([], dtype = torch.long, device='cuda:0')

    def step(self, temperature, dones):

        not_done_indices = [i for i, done in enumerate(dones) if not done]

        inputs_conditioned_padded = pad_sequence(self.input_conditioned, batch_first = True)
        original_lengths = [t.size(0) for t in self.inputs_conditioned]
        attention_mask = inputs_conditioned_padded.ne(0)

        with torch.no_grad():

            outputs = self._generation_model(
                input_ids = inputs_conditioned_padded,
                attention_mask = attention_mask,
                return_dict = True #,
                # output_hidden_state = True
            )


        logits = outputs.logits
        logits = [logits[i,original_lengths[i]-1] for i in range(len(original_lengths))]

        new_dones = [True]*len(dones)
        rewards = [0]*len(dones)
        model_entropies = [0] * len(dones)
        current_words = [""]*len(dones)

        for idx, original_index in enumerate(not_done_indices):

            top_k_values, top_k_indices = -torch.topk(logits[original_index], self.top_k)

            model_distribution = Categorical(logits=top_k_values)
            epsilon = 1e-6
            entropy = -torch.sum(model_distribution.probs * torch.log2(model_distribution.probs + epsilon))
            if math.isnan(entropy):
                print("There is a problem")
            model_entropies[original_index] = entropy.item()

            altered_top_k_values = top_k_values/temperature[original_index]

            distribution = Categorical(logits = altered_top_k_values)
            sampled_index = distribution.sample().item()
            next_token_id = top_k_indices[sampled_index].unsqueeze(0)

            current_word = self._generation_tokenizer.decode([next_token_id.item()])
            current_words[original_index] = current_word

            self.inputs_conditioned[original_index] = torch.cat([self.inputs_conditioned[original_index], next_token_id])

            if next_token_id.item() == self._generation_tokenizer.eos_token_id:
                new_dones[original_index] = True
            else:
                new_dones[original_index] = False

        sentences_returned = [self._generation_tokenizer.decode(seq) for seq in self.inputs_conditioned]

        del outputs, logits, top_k_values, top_k_indices, altered_top_k_values

        return sentences_returned, rewards, new_dones, current_words, model_entropies
    
