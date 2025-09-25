from vllm import LLM, SamplingParams
import random
import os


class VllmAgent:
    def __init__(self, model_name, num_gpus=2, max_tokens=1000, max_model_len=None, tensor_parallel_size=None, **kwargs):
        self.model_name = model_name
        self.model = LLM(
            model=model_name,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=0.9,
            max_logprobs=9999999999,
            max_model_len=max_model_len,  
        )

        self.tokenizer = self.model.get_tokenizer()
        self.max_tokens = max_tokens
        # # Default llama 3 generation config from huggingface
        # generation_config = {
        #     # "bos_token_id": 128000,
        #     # "eos_token_id": [128001, 128009],
        #     "do_sample": True,
        #     "temperature": 0.6,
        #     "max_length": 4096,
        #     "top_p": 0.9,
        #     "transformers_version": "4.40.0.dev0",
        # }
        self.temperature = 0.6
        self.stop_tokens = kwargs.get("stop_tokens", None)

    def preprocess_input(self, text, system_prompt=None):
        prompt_dict = []
        if system_prompt:
            prompt_dict.append({"role": "system", "content": system_prompt})
        elif '[SYS]' in text:
            text = text.split('[SYS]')
            prompt_dict.append({"role": "system", "content": text[0]})
            text = text[-1]

        prompt_broken = text.split("[INST]")
        for idx, val in enumerate(prompt_broken):
            if idx % 2:
                prompt_dict.append({"role": "assistant", "content": val})
            else:
                prompt_dict.append({"role": "user", "content": val})
        # breakpoint()

        prompt = self.tokenizer.apply_chat_template(
            prompt_dict, tokenize=False, add_generation_prompt=True
        )
        # print(prompt)
        return prompt
        # messages = [
        # # {"role": "system", "content": "You are a helpful assistant."},
        #   {"role": "user", "content": text},
        # ]
        # chat_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # return chat_prompt

    def postprocess_output(self, output):
        return output.outputs[0].text.strip()

    def postprocess_output_choice(self, output, choices):
        try:
            probs = list(output.outputs[0].logprobs[0].values())
            probs = sorted(probs, key=lambda x: x.rank)
            for prob in probs:
                if prob.decoded_token in choices:
                    return prob.decoded_token
            return random.choice(choices)
        except:
            return random.choice(choices)

    def interact(
        self, text, temperature=None, max_tokens=None, system_prompt=None, choices=None
    ):
        return self.batch_interact(
            [text],
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            choices=choices,
        )[0]

    def batch_interact(
        self,
        texts,
        temperature=None,
        max_tokens=None,
        stop_tokens=None,
        prompt_logprobs=None,
        system_prompt=None,
        choices=None,
    ):
        if max_tokens is None:
            max_tokens = self.max_tokens
        if temperature is None:
            temperature = self.temperature
        if stop_tokens is None:
            stop_tokens = self.stop_tokens

        if choices is not None:
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=1,
                max_tokens=1,
                stop=stop_tokens,
                logprobs=100,
            )
        else:
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=0.9,
                max_tokens=max_tokens,
                stop=stop_tokens,
                prompt_logprobs=prompt_logprobs,
            )

        prompts = [self.preprocess_input(text, system_prompt) for text in texts]
        outputs = self.model.generate(prompts, sampling_params=sampling_params)

        if choices is not None:
            responses = [
                self.postprocess_output_choice(output, choices) for output in outputs
            ]
        else:
            responses = [self.postprocess_output(output) for output in outputs]

        return responses

    # def batch_compute_likelihood(self, texts, temperature=None, max_tokens=None, stop_tokens=None, prompt_logprobs=None):
    #     if max_tokens is None:
    #         max_tokens = self.max_tokens
    #     if temperature is None:
    #         temperature = self.temperature
    #     if stop_tokens is None:
    #         stop_tokens = self.stop_tokens

    #     sampling_params = SamplingParams(temperature=temperature, top_p=1, max_tokens=max_tokens, stop=self.stop_tokens, prompt_logprobs=prompt_logprobs)
    #     prompts = [self.preprocess_input(text) for text in texts]
    #     outputs = self.model.generate(prompts, sampling_params=sampling_params)

    #     return outputs
    def print_and_reset_usage(self):
        pass


class VllmBaseAgent(VllmAgent):
    def preprocess_input(self, text):
        return text
