from src.utils import *
from rouge_score import rouge_scorer
import numpy as np
from src.load_dataset import load_dataset_from_config
from functools import partial


def process(
    step_conf,
    global_conf,
    dataset,
    access_func,
    output_path,
    print_and_write_results,
    output_dataset_key,
    gpt_method,
    method_name_config,
    dataset_name,
    **kwargs,
):
    import transformers

    # Need to adjust it in the future to deal with finetuned things
    from peft import AutoPeftModelForCausalLM, PeftConfig, PeftModel

    # breakpoint()
    model_path_str = step_conf.method.replace("dpft_sanitize-", "")
    model_root_path = Path(global_conf.dpft_sanitize.model_root_path)

    # model_name_config = dict(
    #     map(lambda x: x.split("_", 1), model_path_str.split("-"))
    # )
    model_name_config = method_name_config
    # breakpoint()
    cue_method = model_name_config["cued"]
    if 'eps' in model_name_config:
        eps = model_name_config["eps"]
    else:
        eps = model_name_config["nm"]
    # Model Loading
    # need to change in the future
    if "no" in model_name_config:
        no_lora = model_name_config["no"] == "lora"
    else:
        no_lora = False
    peft_config = None
    if eps == "0":
        # hardcoded
        # breakpoint()
        model = transformers.AutoModelForCausalLM.from_pretrained("gpt2-large")
    elif no_lora:
        model_path = model_root_path / model_path_str
        model = transformers.AutoModelForCausalLM.from_pretrained(model_path)
    else:
        model_path = model_root_path / model_path_str
        # model = AutoPeftModelForCausalLM.from_pretrained(model_path)
        peft_config = PeftConfig.from_pretrained(model_path)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path
        )
    model.to("cuda")

    # tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2-large")
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "left"

    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2-large")
    tokenizer.padding_side = "left"

    num_added_toks = tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if peft_config is not None or eps == "0":
        mean_tok_emb = model.transformer.wte.weight.data.mean(dim=0)
        model.resize_token_embeddings(len(tokenizer))

        # Initialize the newly-added token embedding to the mean of all token embeddings
        for i in range(num_added_toks):
            model.transformer.wte.weight.data[-(i + 1), :] = mean_tok_emb

    if peft_config is not None:
        model = PeftModel.from_pretrained(model, model_path)

    cue_words_dataset_key = "cue_words"

    def create_cues(entry):
        if cue_method == "specialty":
            fields = ["specialty"]
        elif cue_method == "specialty_subspecialty":
            fields = ["specialty", "subspecialty"]
        elif cue_method == "qa":
            # I'll do this for now...
            if "ddxp" in step_conf.dataset_in.name:
                fields = ["answer"]
            elif "medqa" in step_conf.dataset_in.name:
                fields = ["question", "answer"]
            else:
                raise
        elif cue_method == "none":
            if "wildchat" in step_conf.dataset_in.name:
                fields = []
            else:
                raise
        elif cue_method == "title":
            if "pmc" in step_conf.dataset_in.name:
                fields = ["title"]
            else:
                raise
        else:
            fields = []

        field_values = map(
            lambda field: (
                entry[field] if field in entry else entry["patient"][field] or ""
            ),
            fields,
        )
        text = "\t".join(map(": ".join, zip(fields, field_values)))
        # return {cue_words_dataset_key: text + "\n\n"}
        # Note two \n\n at the end WILL get tokenized into something else
        return {cue_words_dataset_key: text + "\n"}

    dataset = dataset.map(create_cues)

    def generate_text(prompts):
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = inputs.to(model.device)

        # generate up to 30 tokens
        # breakpoint()

        # outputs = model.generate(**inputs, do_sample=True, max_length=512)
        outputs = model.generate(**inputs, **global_conf.dpft_sanitize.sampling_params)

        processed_output = list(
            map(
                lambda cue_words, output: output.replace(cue_words, "").strip(),
                prompts,
                tokenizer.batch_decode(outputs, skip_special_tokens=True),
            )
        )

        return processed_output

    dataset = dataset.map(
        wrap_dataset_map_func(generate_text, cue_words_dataset_key, output_dataset_key),
        batched=True,
        batch_size=8,
    )
    del model
    free_memory()

    set_access_key(dataset, "sanitized_document", output_dataset_key, global_conf)
    return dataset
