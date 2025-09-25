from src.utils import *



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

    if "id" not in dataset.column_names:
        if 'wildchat' in dataset_name:
            dataset = dataset.rename_column("Unnamed: 0", "id")
        else:
            raise
    else:
        return

    return dataset