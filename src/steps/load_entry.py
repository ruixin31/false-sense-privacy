from src.utils import *
from src.load_dataset import load_dataset_from_config
from datasets import load_from_disk

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
    dataset_name_in_conf = step_conf.extra_config.get('loadds', None)
    extra_config = step_conf.extra_config.get('configs', None)
    if extra_config:
        to_load_dataset_config = step_conf.dataset_out.copy()
        for key, value in to_load_dataset_config.items():
            if isinstance(value, str):
                for replacement in extra_config.get('replacements', []):
                    value = value.replace(replacement[0], replacement[1])
                to_load_dataset_config[key] = value
    else:
        to_load_dataset_config = global_conf.get(dataset_name_in_conf)
    to_load_dataset, to_load_access_func = load_dataset_from_config(to_load_dataset_config)
    # get the facts field, and set it to the current dataset
    def process(entries):
        assert entries['id'] == to_load_dataset['id']
        entries[output_dataset_key] = to_load_access_func(to_load_dataset)
        return entries
    dataset = dataset.map(process, batched=True, batch_size=999999999999)
    
    if extra_config:
        set_access_key(dataset, "sanitized_document", output_dataset_key, global_conf)

    return dataset
