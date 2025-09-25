from src.utils import *
import os
import random
import json

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
    
    idx_selection = json.loads(get_access_key(dataset, "randomized_idxes", global_conf))


    set_rand3_idx_from_idx_selection(idx_selection, global_conf)    


    return dataset