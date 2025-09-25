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

    if "wildchat" in dataset_name:
        adv_inf_instr = "Let us play a guessing game. The following is the a conversation between a user and an AI chatbot. Your task is to guess the user's attributes including Age, Education, Income, Location, Occupation, Relationship Status, Sex, Place of Birth?"
        data_type_sanitization = "conversation between a user and an AI chatbot"
        data_type_word = "conversation"
    else:
        adv_inf_instr = "Let us play a guessing game. The following is a synthetic patient record. Your task is to guess the patient's attributes including Age, Education, Income, Location, Occupation, Relationship Status, Sex, Place of Birth?"
        data_type_sanitization = "medical record from a user"
        data_type_word = "record"

    
    user_prompt = f"""You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible. Only generalize information and do not invent new information.
Example: 'my husband and I' -> 'my partner and I' is valid, but 'my husband and I' -> 'my wife and I' is not.
Example: 'my husband and I have a dog' -> 'my partner and I have a dog' is valid, but 'my husband and I have a dog'
-> 'my partner and I have a cat' is not.
Example: 'my husband and I' -> 'I' is also valid as it only removes information.
Below I give you a {data_type_sanitization}. Your task is to anonymize the {data_type_word} to prevent these inferences, while keeping the rest of the text intact.  
    
{data_type_word.title()}: 
{{}}

In a first step briefly explain the changes you will make. In a new line write a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information instead generalize information. Do not add indicators besides #, i.e. do not add phrases such as Anonymized text: or similar."""

    if method_name_config.get('use') == 'gpt':
        local_gpt_method = LM("gpt-4o-2024-08-06")
        local_gpt_method.load_model()
        # Per their paper
        local_gpt_method.model.temperature = 0.1
        gpt_method = local_gpt_method

    intermediate_key = "sanitized_document"
    prompt = Prompt(user_prompt, post_process_func=post_process_sanitize)
    dataset = prompt.apply(dataset, gpt_method, access_func, intermediate_key)
    
    user_prompt_paraphrase = """Given the following text, generate a paraphrased version that maintains the original meaning, context, and tone while using different words and sentence structures. Ensure that the paraphrased text is clear, coherent, and logically organized.

text: {}

In a new line return the anonymized text. Make sure you keep the original meaning, i.e. do not invent new information instead generalize information. Do not add any indicators, i.e. do not add phrases such as paraphrased text: or similar."""

    prompt = Prompt(user_prompt_paraphrase, post_process_func=lambda x: x.replace('#', '').strip())
    dataset = prompt.apply(dataset, gpt_method, intermediate_key, output_dataset_key)

    try:
        local_gpt_method.model.print_and_reset_usage()
    except:
        pass

    # breakpoint()
    set_access_key(dataset, "sanitized_document", output_dataset_key, global_conf)

    return dataset
