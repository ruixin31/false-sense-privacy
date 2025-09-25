from src.utils import *
from datasets import load_from_disk


def post_process_sanitize(sanitized_text):
    try:
        if isinstance(sanitized_text, str):
            finalized_sanitized_text = sanitized_text
        else:
            finalized_sanitized_text = sanitized_text[0]

        finalized_sanitized_text = "#".join(finalized_sanitized_text.split("#")[1:]).strip()

    except Exception as ex:
        print(ex)
        print("Failed to parse as jsonl")
        print(sanitized_text)
        finalized_sanitized_text = None
    return finalized_sanitized_text


def process(
    step_conf,
    global_conf,
    dataset,
    access_func,
    output_path,
    print_and_write_results,
    print_and_write_results_json,
    output_dataset_key,
    gpt_method,
    method_name_config,
    dataset_name,
    **kwargs,
):
    #         system_prompt = """You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible. Only generalize information and do not invent new information.
    # Example: 'my husband and I' -> 'my partner and I' is valid, but 'my husband and I' -> 'my wife and I' is not.
    # Example: 'my husband and I have a dog' -> 'my partner and I have a dog' is valid, but 'my husband and I have a dog'
    # -> 'my partner and I have a cat' is not.
    # Example: 'my husband and I' -> 'I' is also valid as it only removes information."""
    #         user_prompt = """Below I give you a medical record from a user. Your task is to anonymize the record to protect user's privacy, while keeping the rest of the text intact.
#     user_prompt = """You are an expert anonymizer and paraphraser that carefully adapts small parts of the text to make it anonymous and rewords the text to convey the same meaning in a different way. You follow the instructions and format precisely, making sure to change relevant parts to protect privacy while also paraphrasing the rest of the text. Generalize information rather than inventing new details, and maintain the original intent of the text.
# Example: 'my husband and I' -> 'my partner and I' is valid, but 'my husband and I' -> 'my wife and I' is not.
# Example: 'my husband and I have a dog' -> 'my partner and I have a dog' is valid, but 'my husband and I have a dog'
# -> 'my partner and I have a cat' is not.
# Example: 'my husband and I' -> 'I' is also valid as it only removes information.
# Below I give you a medical record from a user. Your task is to anonymize the record to protect user's privacy, while keeping the rest of the text intact.
# Record:
# {}
#
# In a first step briefly explain the changes you will make. In a new line write a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information instead generalize information. Do not add indicators besides #, i.e. do not add phrases such as Anonymized text: or similar."""
#     user_prompt = """You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible. Only generalize information and do not invent new information.
# Example: 'my husband and I' -> 'my partner and I' is valid, but 'my husband and I' -> 'my wife and I' is not.
# Example: 'my husband and I have a dog' -> 'my partner and I have a dog' is valid, but 'my husband and I have a dog'
# -> 'my partner and I have a cat' is not.
# Example: 'my husband and I' -> 'I' is also valid as it only removes information.
# Below I give you a medical record from a user. Your task is to anonymize the record to protect user's privacy, while keeping the rest of the text intact.
# Record:
# {}
#
# In a first step briefly explain the changes you will make. In a new line write a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information instead generalize information. Do not add indicators besides #, i.e. do not add phrases such as Anonymized text: or similar."""

    # intermediate_key = "sanitized_document"
    # prompt = Prompt(user_prompt, post_process_func=post_process_sanitize)
    # dataset = prompt.apply(dataset, gpt_method, access_func, intermediate_key)
    

        # dataset_orig_conf = conf.dataset_in.copy()
        # omegaconf.OmegaConf.set_struct(dataset_orig_conf, False)
        # dataset_orig_conf.path = "data/all_train_specialty_llama2_70b_improved_specialty_2_redone_facts_2.jsonl"
        # dataset_orig_conf.load_method = "load_datasets_json"
        # dataset_orig_conf.split_arg = "train"
        # dataset_orig, _ = load_dataset_from_config(dataset_orig_conf)
        # dataset = dataset.map(
        #     wrap_dataset_map_func(
        #         lambda x: dataset_orig[x]["facts"],
        #         "id",
        #         "facts",
        #     ),
        # )

    
    
    # dataset = dataset.map(
    #     wrap_dataset_map_func(
    #         lambda x: list(dict.fromkeys(x)), access_func, output_dataset_key
    #     ),
    # )
    # set_access_key(dataset, "sanitized_facts", output_dataset_key, global_conf)

    # Note that the baseline here means that changes that affect all runs
    step_speical_column_name = 'paraphrased_original_facts'
    if global_conf.is_baseline or global_conf.is_v2_run:
        user_prompt_paraphrase = """Given the following text, generate a paraphrased version that maintains the original meaning, context, and tone while using different words and sentence structures. Ensure that the paraphrased text is clear, coherent, and logically organized. 

text: {}

In a new line write a single # and then return the paraphrased text. Make sure you keep the original meaning, i.e. do not invent new information instead generalize information. Do not add indicators besides #, i.e. do not add phrases such as Paraphrased text: or similar."""

        # breakpoint()
        # access_func = get_access_key(dataset, "facts", global_conf)

        def get_aux(entry):
            field_of_concern = entry[access_func]
            if not len(field_of_concern) == 1:
                return ''
            assert len(field_of_concern) == 1
            return field_of_concern[0]
        prompt = PromptWithCache(
            output_path,
                gpt_method,
            set_alt_job_for_inference=global_conf.set_alt_job_for_inference,
            exp_id=global_conf.transform_dataset.exp_folder,
            bin_id="paraphrase_facts_v1",
        )
        while prompt.should_keep_looping():
            output = []
            for entry in dataset: 
                output.append([])
                facts = access_func(entry)
                paraphrase_prompts = list(map(lambda x: user_prompt_paraphrase.format(x), facts))
                paraphrase_prompts_res = list(map(prompt.prompt, paraphrase_prompts))
                if not paraphrase_prompts_res or paraphrase_prompts_res[0] is None:
                    continue
                
                processed = list(map(post_process_sanitize, paraphrase_prompts_res))
                output[-1] = processed


        if prompt.resolved:
            dataset = dataset.add_column(step_speical_column_name, output)
            set_access_key(dataset, "matching_cues", step_speical_column_name, global_conf)
            return dataset

    else:
        if step_speical_column_name not in dataset.column_names:
            model = global_conf.model.lm_model
            if global_conf.gpt_exps_unique_load_facts_when_paraphrasing:
                if "medqa" in dataset_name:
                    facts_paraphrased_dataset = f'{global_conf.transform_dataset.exp_folder}/baseline_1k-medqa_factorized-{model}-1000/paraphrase_facts-facts-medqa_factorized-{model}-1000/dataset'
                elif "wildchat" in dataset_name:
                    facts_paraphrased_dataset = f'{global_conf.transform_dataset.exp_folder}/baseline_1k-wildchat_all_unprocessed-{model}-1000/paraphrase_facts-facts-wildchat_all_unprocessed-{model}-1000/dataset'
                else:
                    raise
            else:
                if "medqa" in dataset_name:
                    facts_paraphrased_dataset = f'{global_conf.transform_dataset.exp_folder}/baseline_1k-medqa_factorized-{model}-1000/paraphrase_facts-medqa_factorized-{model}-1000/dataset'
                elif "wildchat" in dataset_name:
                    facts_paraphrased_dataset = f'{global_conf.transform_dataset.exp_folder}/baseline_1k-wildchat_all_unprocessed-{model}-1000/paraphrase_facts-wildchat_all_unprocessed-{model}-1000/dataset'
                else:
                    raise
            facts_paraphrased_dataset = load_from_disk(facts_paraphrased_dataset)
            access_func = get_access_key(facts_paraphrased_dataset, "matching_cues", global_conf)
            assert dataset['id'] == facts_paraphrased_dataset['id']

            dataset = dataset.add_column(step_speical_column_name, facts_paraphrased_dataset[access_func])

            set_access_key(dataset, "matching_cues", step_speical_column_name, global_conf)
            if global_conf.gpt_exps_unique_load_facts_when_paraphrasing:
                access_func = get_access_key(facts_paraphrased_dataset, "sanitized_facts", global_conf)
                dataset = dataset.remove_columns('facts')
                dataset = dataset.add_column('facts', facts_paraphrased_dataset[access_func])

            #     dataset = dataset.map(
            #         wrap_dataset_map_func(
            #             lambda x: x[access_func],
            #             "id",
            #             step_speical_column_name,

            return dataset
        else:
            return
