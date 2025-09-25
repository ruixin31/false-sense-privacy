from src.utils import *
from src.gpt import LM


def post_process_sanitize(sanitized_text):
    try:
        finalized_sanitized_text = "#".join(sanitized_text.split("#")[1:]).strip()

    except Exception as ex:
        print(ex)
        print("Failed to parse as jsonl")
        print(sanitized_text)
        finalized_sanitized_text = None
    return finalized_sanitized_text


## TODO: Classify the text as medical or chatbot


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
    # local_gpt_method = LM("gpt-4-1106-preview")
    local_gpt_method = LM("gpt-4o-2024-08-06")
    local_gpt_method.load_model()
    # Per their paper
    local_gpt_method.model.temperature = 0.1
    # local_gpt_method = gpt_method
    if "wildchat" in dataset_name:
        adv_inf_instr = "Let us play a guessing game. The following is the a conversation between a user and an AI chatbot. Your task is to guess the user's attributes including Age, Education, Income, Location, Occupation, Relationship Status, Sex, Place of Birth?"
        data_type_sanitization = "conversation between a user and an AI chatbot"
        data_type_word = "conversation"
    else:
        adv_inf_instr = "Let us play a guessing game. The following is a synthetic patient record. Your task is to guess the patient's attributes including Age, Education, Income, Location, Occupation, Relationship Status, Sex, Place of Birth?"
        data_type_sanitization = "medical record from a user"
        data_type_word = "record"

    def generate_format_sanitization(record_key, inference_key):
        # breakpoint()
        if isinstance(record_key, str):
            record_key_func = lambda x: x[record_key]
        else:
            record_key_func = record_key

        def format_inference(entry):
            return record_key_func(entry) + "\n\n\nInference that can be made from the user (do not include the below in the sanitization): \n" + entry[inference_key]

        return format_inference

    adv_inf_system_prompt = "You are an expert investigator with years of experience in text analysis. You work with an analytical mindset and try to answer questions as precisely as possible."
    adv_inf_user_prompt = f"""{adv_inf_instr} Only include attributes that might be inferred from the text.

Type: <target_attribute>
Inference: Your detailed inference  
Guess: Your top 3 guesses in a list separated by ; 
Certainty: Based on your inference how certain you are from 1-5. 1 is very uncertain only based on biases and 5 has clear evidence in the {data_type_word}. Only give the number.

{data_type_word.title()}: 
{{}}"""

    sanitizaiton_system_prmopt = """You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible. Only generalize information and do not invent new information.  

    Example: 'my husband and I' -> 'my partner and I' is valid, but 'my husband and I' -> 'my wife and I' is not. 
    Example: 'my husband and I have a dog' -> 'my partner and I have a dog' is valid, but 'my husband and I have a dog' -> 'my partner and I have a cat' is not. 
    Example: 'my husband and I' -> 'I' is also valid as it only removes information."""

    sanitizaiton_user_prmopt = f"""Below I give you a {data_type_sanitization} as well as personal inferences made about the user that can be made from the {data_type_word}. Your task is to anonymize the {data_type_word} to prevent these inferences, while keeping the rest of the text intact.  
    
{data_type_word.title()}: 
{{}}

In a first step briefly explain the changes you will make. In a new line write a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information instead generalize information.
"""

    record_key = access_func
    for turn in range(3):
    # for turn in range(1):
        turn_str = str(turn + 1)
        adv_inf_key = f"adv_inf_{turn_str}"
        sanitization_key = f"sanitization_{turn_str}"
        adv_inf_prompt = Prompt(adv_inf_user_prompt, adv_inf_system_prompt)
        dataset = adv_inf_prompt.apply(
            dataset,
            local_gpt_method,
            record_key,
            adv_inf_key,
            processing_args={"num_proc": 6, "batch_size": 1, "batched": True},
        )

        sanitization_prompt = Prompt(
            sanitizaiton_user_prmopt,
            sanitizaiton_system_prmopt,
            post_process_func=post_process_sanitize,
        )
        dataset = sanitization_prompt.apply(
            dataset,
            local_gpt_method,
            generate_format_sanitization(record_key, adv_inf_key),
            sanitization_key,
            processing_args={"num_proc": 6, "batch_size": 1, "batched": True},
        )

        record_key = sanitization_key

    # breakpoint()
    dataset = dataset.rename_columns({sanitization_key: output_dataset_key})

    try:
        local_gpt_method.model.print_and_reset_usage()
    except:
        pass

    #     intermediate_key = "sanitized_document"
    #     prompt = Prompt(user_prompt, post_process_func=post_process_sanitize)
    #     dataset = prompt.apply(dataset, gpt_method, access_func, intermediate_key)

    #     prompt = Prompt("{}")
    #     dataset = prompt.apply(dataset, gpt_method, format_prompt, output_dataset_key)

    #     user_prompt_paraphrase = """Given the following text, generate a paraphrased version that maintains the original meaning, context, and tone while using different words and sentence structures. Ensure that the paraphrased text is clear, coherent, and logically organized.

    # text: {}

    # In a new line return the anonymized text. Make sure you keep the original meaning, i.e. do not invent new information instead generalize information. Do not add any indicators, i.e. do not add phrases such as paraphrased text: or similar."""

    #     prompt = Prompt(user_prompt_paraphrase, post_process_func=lambda x: x.replace('#', '').strip())
    #     dataset = prompt.apply(dataset, gpt_method, intermediate_key, output_dataset_key)

    # breakpoint()
    set_access_key(dataset, "sanitized_document", output_dataset_key, global_conf)

    return dataset
