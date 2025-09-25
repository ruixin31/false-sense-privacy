from src.utils import *
import os
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
    
    access_key = get_access_key(dataset, "matching_cues", global_conf)

    output = {}

    random.seed(42)

    for record in dataset:
        all_idxs = list(range(len(record[access_key])))
        random.shuffle(all_idxs)
        output[record['id']] = all_idxs[:3]
    set_access_key(dataset, "randomized_idxes", json.dumps(output), global_conf)



    return dataset