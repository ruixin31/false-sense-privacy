# import os
# import pickle
# import time

# import torch
# from torch.nn.utils.rnn import pad_sequence
# import tqdm
# import transformers
# from transformers import AutoModelForCausalLM, AutoTokenizer

# # I'm going to create a bunch of helper functions to do this
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# class MistralModel():
#     def __init__(self, model_name, cache_file=".mistral-7B-Instruct-v0.2.cache"):
#         self.model_name = model_name
#         self.model = None
#         # TODO: wire this up - currently the cache is unused
#         self.cache_file = cache_file
#         self.cache_dict = self.load_cache()

#     def load_model(self):
#         print("Mistral load_model called!")
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.model_name,
#             # cache_dir="/gscratch/xlab/blnewman/models/transformers/",
#             # load_in_8bit=True,
#             torch_dtype=torch.float16,
#             device_map="auto",
#         )
#         self.model.eval()
#         self.model.config.pad_token_id = self.model.config.eos_token_id
#         # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir="/gscratch/xlab/blnewman/models/transformers/")
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#         self.tokenizer.pad_token = self.tokenizer.eos_token
    
#     def generate(self, prompt, max_output_length=1000, do_sample=True, **kwargs):
#         return self.generate_batch([prompt], max_output_length=max_output_length, do_sample=do_sample, **kwargs)
#         # if self.model is None:
#         #     self.load_model()
#         # transformers.set_seed(0)
#         # prompt_dict = [{"role": "user", "content": prompt}]
#         # inputs = self.tokenizer.apply_chat_template(prompt_dict, return_tensors="pt").to("cuda")
#         # generated_ids = self.model.generate(
#         #     inputs,
#         #     max_new_tokens=max_output_length,
#         #     do_sample=do_sample,
#         #     num_return_sequences=1,
#         #     **kwargs
#         # )
    
#         # new_tokens_batch = self.tokenizer.batch_decode(
#         #     generated_ids[:, inputs.shape[1]:],
#         #     skip_special_tokens=True
#         # )

#         # return new_tokens_batch

#     def generate_batch(self, prompts_batch, batch_size=4, max_output_length=1000, do_sample=True, **kwargs):
#         if self.model is None:
#             self.load_model()
#         inputs = []
#         for prompt in prompts_batch:
#             prompt_dict = [{"role": "user", "content": prompt}]
#             inputs.append(
#                 self.tokenizer.apply_chat_template(
#                     prompt_dict, return_tensors="pt",# padding="max_length", max_length=55
#                 ).to(DEVICE)
#             )
        
#         # pad sequences on the left (hence the flipping)
#         # print("[generate_batch], len(inputs)", len(inputs))
#         inputs = pad_sequence([
#             inpt.flatten().flip(dims=[0]) for inpt in inputs
#         ], batch_first=True, padding_value=self.tokenizer.pad_token_id).flip(dims=[1])
#         attn_mask = (inputs != self.tokenizer.pad_token_id)

#         generations_batch = []
#         # if batch_size is None:
#         #     batch_size = len(prompts_batch)

#         for i in range(0, len(inputs), batch_size):
#             transformers.set_seed(0)
            
#             generated_ids = self.model.generate(
#                 inputs[i:i+batch_size],
#                 attention_mask=attn_mask[i:i+batch_size],
#                 max_new_tokens=max_output_length,
#                 do_sample=do_sample,
#                 num_return_sequences=1,
#                 pad_token_id=self.tokenizer.eos_token_id,
#                 **kwargs
#             ).cpu()
#             generations_batch.extend(
#                 self.tokenizer.batch_decode(
#                     generated_ids[:, inputs.shape[1]:],
#                     skip_special_tokens=True
#                 )
#             )
#         inputs.to("cpu")
#         attn_mask.to("cpu")
#         return generations_batch

#     def save_cache(self):
#         # if self.add_n == 0:
#         #     return

#         # load the latest cache first, since if there were other processes running in parallel, cache might have been updated
#         for k, v in self.load_cache().items():
#             self.cache_dict[k] = v

#         with open(self.cache_file, "wb") as f:
#             pickle.dump(self.cache_dict, f)

#     def load_cache(self, allow_retry=True):
#         if os.path.exists(self.cache_file):
#             while True:
#                 try:
#                     with open(self.cache_file, "rb") as f:
#                         cache = pickle.load(f)
#                     break
#                 except Exception:
#                     if not allow_retry:
#                         assert False
#                     print ("Pickle Error: Retry in 5sec...")
#                     time.sleep(5)        
#         else:
#             cache = {}
#         return cache


# MISTRAL = MistralModel("mistralai/Mistral-7B-Instruct-v0.2")

import os
import pickle
import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

token = os.environ['HF_API_KEY']

# I'm going to create a bunch of helper functions to do this
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CausalLM:
    def __init__(self, model_name, cache_file=None):
        self.model_name = model_name
        self.model = None
        # TODO: wire this up - currently the cache is unused
        if cache_file is not None:
            self.cache_file = cache_file
            self.cache_dict = self.load_cache()

    def load_model(self):
        print("CausalLM load_model called!")
        if "meta-llama/Meta-Llama-3-8B-Instruct" in self.model_name:
            
            pipeline = transformers.pipeline(
                "text-generation",
                model=self.model_name,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device="cuda",
                token=token,
            )
            self.model = pipeline
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                # cache_dir="/gscratch/xlab/blnewman/models/transformers/",
                # load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map="auto",
                token=token,
            )
            self.model.eval()
            self.model.config.pad_token_id = self.model.config.eos_token_id
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, #cache_dir="/gscratch/xlab/blnewman/models/transformers/"
                token=token,
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if "llama-2-7b-chat-hf" in self.model_name or "llama-2-13b-chat-hf" in self.model_name:
                self.tokenizer.padding_side = "left"

    def generate(self, prompt, max_output_length=1000, do_sample=True, system_prompt=None,  **kwargs):
        if "meta-llama/Meta-Llama-3-8B-Instruct" in self.model_name:
            if self.model is None:
                self.load_model()
            # breakpoint()
            pipeline = self.model
            prompt_dict = []
            if system_prompt:
                prompt_dict.append({"role": "system", "content": system_prompt})

            prompt_broken = prompt.split('[INST]')
            for idx, val in enumerate(prompt_broken):
                if idx % 2:
                    prompt_dict.append({"role": "assistant", "content": val})
                else:
                    prompt_dict.append({"role": "user", "content": val})
            # breakpoint()
                
            prompt = self.model.tokenizer.apply_chat_template(
                    prompt_dict, 
                    tokenize=False, 
                    add_generation_prompt=True
            )

            terminators = [
                pipeline.tokenizer.eos_token_id,
                pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = pipeline(
                prompt,
                max_new_tokens=1000,
                eos_token_id=terminators,
                do_sample=do_sample,
                # temperature=0,
                # top_p=1,
            )
            return [outputs[0]["generated_text"][len(prompt):]]

        return self.generate_batch(
            [prompt],
            batch_size=1,
            max_output_length=max_output_length,
            do_sample=do_sample,
            **kwargs,
        )

    def generate_batch(
        self,
        prompts_batch,
        batch_size=4,
        max_output_length=1000,
        do_sample=True,
        num_return_sequences=1,
        output_scores=False,
        **kwargs,
    ):
        if self.model is None:
            self.load_model()
        inputs = []
            
        if self.model_name == "mistralai/Mistral-7B-Instruct-v0.2":
            for prompt in prompts_batch:
                prompt_dict = [{"role": "user", "content": prompt}]
                inputs.append(
                    self.tokenizer.apply_chat_template(
                        prompt_dict,
                        return_tensors="pt",  # padding="max_length", max_length=55
                    ).to(DEVICE)
                )
            # pad sequences on the left (hence the flipping)
            # print("[generate_batch], len(inputs)", len(inputs))
            inputs = pad_sequence(
                [inpt.flatten().flip(dims=[0]) for inpt in inputs],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            ).flip(dims=[1])
            attn_mask = inputs != self.tokenizer.pad_token_id

        elif "llama-2-7b-chat-hf" in self.model_name or "llama-2-13b-chat-hf" in self.model_name:
            prompts_batch = [f"[INST] {prompt} [/INST]" for prompt in prompts_batch]
            inputs = self.tokenizer(prompts_batch, return_tensors="pt", padding=True).to(DEVICE)
            inputs = inputs.input_ids
            attn_mask = None
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")
        generations_batch = []
        logits_batch = []
        # if batch_size is None:
        #     batch_size = len(prompts_batch)

        for i in range(0, len(inputs), batch_size):
            transformers.set_seed(0)

            generated_output = self.model.generate(
                input_ids=inputs[i : i + batch_size],
                attention_mask=(attn_mask[i : i + batch_size] if attn_mask is not None else None),
                max_new_tokens=max_output_length,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                return_dict_in_generate=True,
                output_scores=output_scores,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )
            generated_ids = generated_output["sequences"].cpu()
            if output_scores:
                logits_misformatted = np.array(
                    [logit.detach().cpu().numpy() for logit in generated_output["scores"]]
                )
                logits_batch.extend([x for x in logits_misformatted.transpose(1, 0, 2)])

            generations_batch.extend(
                self.tokenizer.batch_decode(generated_ids[:, inputs.shape[1] :], skip_special_tokens=True)
            )
        inputs.to("cpu")
        if attn_mask is not None:
            attn_mask.to("cpu")

        if output_scores:
            return generations_batch, logits_batch
        else:
            return generations_batch

    def save_cache(self):
        # if self.add_n == 0:
        #     return

        # load the latest cache first, since if there were other processes running in parallel, cache might have been updated
        for k, v in self.load_cache().items():
            self.cache_dict[k] = v

        with open(self.cache_file, "wb") as f:
            pickle.dump(self.cache_dict, f)

    def load_cache(self, allow_retry=True):
        if os.path.exists(self.cache_file):
            while True:
                try:
                    with open(self.cache_file, "rb") as f:
                        cache = pickle.load(f)
                    break
                except Exception:
                    if not allow_retry:
                        assert False
                    print("Pickle Error: Retry in 5sec...")
                    time.sleep(5)
        else:
            cache = {}
        return cache


MISTRAL = CausalLM("mistralai/Mistral-7B-Instruct-v0.2")

LLAMA_3_CHAT = CausalLM("meta-llama/Meta-Llama-3-8B-Instruct")
# LLAMA_2_13B_CHAT = CausalLM("/gscratch/xlab/blnewman/models/llama/llama-2-13b-chat-hf/")