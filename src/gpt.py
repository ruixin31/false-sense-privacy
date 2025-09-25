import os
import json
import tqdm
import time
import openai
from types import SimpleNamespace
from src.utils import free_memory
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import md5

SYSTEM_PROMPT = f"You are a helpful assistant."


def generate_hash_from_message(message):
    text = ""
    for turn in message:
        text += turn["role"]
        text += "|||"
        text += turn["content"]
        text += "||||"
    return md5(text.encode()).hexdigest()

def generate_hash_from_templates(tasks):
    hashes = []
    for task in tasks:
        hashes.append(task["custom_id"])
    hashes = sorted(hashes)
    text = ""
    for h in hashes:
        text += h
    return md5(text.encode()).hexdigest()

class GPTAgent:
    def __init__(self, kwargs: dict):
        self.args = SimpleNamespace(**kwargs)
        self.temperature = None
        self.usage = {
            "input_tokens": 0,
            "output_tokens": 0,
        }

    def generate_template(self, user_content, system_prompt):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        task = {
            "custom_id": generate_hash_from_message(messages),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                # This is what you would have in your Chat Completions API call
                "model": self.args.model,
                "messages": messages,
            },
        }
        return task
    
    def batch_interact_template(self, user_contents, system_prompt, choices=None, response_file=None):
        tasks = []
        for user_content in user_contents:
            task = self.generate_template(user_content, system_prompt)
            tasks.append(task)

        if response_file is not None:
            results_dict = {}
            with open(response_file, 'r') as file:
                for line in file:
                    # Parsing the JSON string into a dict and appending to the list of results
                    json_object = json.loads(line.strip())
                    results_dict[json_object["custom_id"]] = json_object['response']['body']['choices'][0]['message']['content']

            outputs = []
            for task in tasks:
                if task["custom_id"] not in results_dict:
                    outputs.append("ERROR: Task not found in response file")
                else:
                    outputs.append(results_dict[task["custom_id"]])
            return outputs


        file_name = f"playground/templates/{generate_hash_from_templates(tasks)}.jsonl"
        with open(file_name, 'w') as file:
            for obj in tasks:
                file.write(json.dumps(obj) + '\n')

    def _generate(self, user_content, system_prompt):
        trials = 0
        while True:
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_content})

                completion = openai.chat.completions.create(
                    model=self.args.model,
                    messages=messages,
                    temperature=self.temperature
                )
                # breakpoint()
                self.usage["input_tokens"] += completion.usage.prompt_tokens
                self.usage["output_tokens"] += completion.usage.completion_tokens
                break
            except (openai.APIError, openai.RateLimitError) as e:
                trials += 1
                if trials >= 100:
                    return None
                print("Error: {}".format(e))
                time.sleep(2)
                continue

        return completion

    def interact(self, user_content, system_prompt=None, choices=None):
        response = self._generate(user_content, system_prompt)
        if response:
            output = self.parse_basic_text(response)
        else:
            output = "ERROR: No response"

        return output

    def batch_interact(self, user_contents, system_prompt, choices=None):
        # outputs = []
        # for user_content in tqdm.tqdm(user_contents):
        #     response = self.interact(user_content, system_prompt)
        #     outputs.append(response)
        # return outputs
        # breakpoint()
        outputs = [None] * len(user_contents)
        with ThreadPoolExecutor(max_workers=min(128, len(user_contents))) as executor:
            future_to_idx = {
                executor.submit(self.interact, user_content, system_prompt): idx 
                for idx, user_content in enumerate(user_contents)
            }
            
            for future in tqdm.tqdm(as_completed(future_to_idx), total=len(user_contents)):
                idx = future_to_idx[future]
                try:
                    outputs[idx] = future.result()
                except Exception as e:
                    outputs[idx] = f"ERROR: {str(e)}"
                    
        return outputs

    def parse_basic_text(self, response):
        output = response.choices[0].message.content.strip()

        return output

    def print_and_reset_usage(self):
        print("Input tokens: {}".format(self.usage["input_tokens"]))
        print("Output tokens: {}".format(self.usage["output_tokens"]))
        self.usage = {
            "input_tokens": 0,
            "output_tokens": 0,
        }

class LM:
    def __init__(self, model="gpt-4o-2024-08-06"):
        self.model_name = model
        self.model = None

    def unload_model(self):
        if self.model is not None:
            if "llama" in self.model_name:
                from vllm.distributed.parallel_state import destroy_model_parallel

                destroy_model_parallel()
                del self.model.model.llm_engine.model_executor.driver_worker

            self.model = None
            free_memory()

    def load_model(self):
        model = self.model_name
        if "gpt-4" in model or "turbo" in model:
            gpt = GPTAgent(
                {
                    "model": model,
                    # "temperature": 0,
                    # "top_p": 1.0,
                    # "frequency_penalty": 0.0,
                    # "presence_penalty": 0.0,
                }
            )
        elif "mistral" in model:
            from src.mistral import MISTRAL

            gpt = MISTRAL
            gpt.interact = lambda user_content, system_prompt: gpt.generate(
                user_content
            )[0]
        elif "llama" in model:
            from .vllm_wrapper import VllmAgent, VllmBaseAgent

            additional_args = {}
            num_gpus = 1
            if model == "llama":
                model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
            elif model == "llama-3.1":
                model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            elif model == "llama-3.1-70b":
                model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct"
                additional_args["max_model_len"] = 16384
                num_gpus = 4
            gpt = VllmAgent(
                model_name,
                # "meta-llama/Meta-Llama-3-8B-Instruct",
                # "meta-llama/Meta-Llama-3.1-8B-Instruct",
                num_gpus=num_gpus,
                max_tokens=4096,
                stop_tokens=["<|eot_id|>", "<|end_of_text|>"],
                **additional_args,
            )

            # gpt.interact = lambda user_content, system_prompt: gpt.interact(
            #     user_content, system_prompt=system_prompt
            # )

            # from src.mistral import LLAMA_3_CHAT

            # gpt = LLAMA_3_CHAT
            # gpt.interact = lambda user_content, system_prompt: gpt.generate(
            #     user_content, do_sample=False, system_prompt=system_prompt
            # )[0]
        else:
            raise
        self.model = gpt

    def prompt_batch(
        self, user_contents, system_prompt="", output_txt=None, choices=None
    ):
        if self.model is None:
            self.load_model()
        output = self.model.batch_interact(
            user_contents, system_prompt=system_prompt, choices=choices
        )
        if output_txt:
            with open(output_txt, "a") as f:
                for user_content, response in zip(user_contents, output):
                    f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")
                    f.write(self.model_name + "\n")
                    f.write(system_prompt + "\n")
                    f.write(user_content + "\n")
                    f.write(
                        "****************************************************************************\n\n"
                    )
                    f.write(response)
                    f.write(
                        "\n=============================================================================\n\n"
                    )
        return output

    def prompt(self, user_content="", system_prompt="", output_txt=None, choices=None):
        if self.model is None:
            self.load_model()
        output = self.model.interact(
            user_content, system_prompt=system_prompt, choices=choices
        )

        if output_txt:
            with open(output_txt, "a") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")
                f.write(self.model_name + "\n")
                f.write(system_prompt + "\n")
                f.write(user_content + "\n")
                f.write(
                    "****************************************************************************\n\n"
                )
                f.write(output)
                f.write(
                    "\n=============================================================================\n\n"
                )
        return output
