from src.utils import *
import random


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

    dataset_context = access_func(dataset)
    
    random.shuffle(dataset_context)

    dataset = dataset.add_column(output_dataset_key, dataset_context)


    set_access_key(dataset, "sanitized_document", output_dataset_key, global_conf)

    return dataset