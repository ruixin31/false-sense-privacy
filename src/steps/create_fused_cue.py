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

    # facts_key_orig = get_access_key(dataset, "original_facts", global_conf)
    facts_key_orig = get_access_key(dataset, global_conf.evaluation.privacy.samples_key_from_orig, global_conf)
    context_key_sanitized = get_access_key(dataset, "sanitized_document", global_conf)
    range_of_interest_str = parse_create_fused_cue_range(
        method_name_config["r"], global_conf
    )
    assert type(range_of_interest_str) == str

    slice_generator_func = lambda y: (
        slice(*(map(lambda x: int(x) if x else None, y.split(":")))) if ":" in y else y
    )

    ranges = list(map(slice_generator_func, [range_of_interest_str]))

    range_of_interest = ranges[0]

    key = f'fused_facts_{method_name_config["r"]}'

    def fuse_cue(entry):
        selected_facts = proper_slice_array_by_range(range_of_interest, entry, entry[facts_key_orig])

        output = " ".join(selected_facts)
        out = None
        if not output:
            out = []
        else:
            out = [output]
        return {
            key: out,
        }

    # dataset = dataset.map(wrap_dataset_map_func(fuse_cue, facts_key_orig, key))
    dataset = dataset.map(fuse_cue)
    set_access_key(dataset, "matching_cues", key, global_conf)

    dataset = dataset.map(
        wrap_dataset_map_func(lambda x: [x], context_key_sanitized, output_dataset_key)
    )
    set_access_key(dataset, "matching_document", output_dataset_key, global_conf)

    # global_conf.range_maps = [":1"]
    # global_conf.evaluation.privacy.samples_key_from_orig = "matching_cues"
    # global_conf.evaluation.privacy.samples_key_from_sanitized = "matching_document"

    # breakpoint()
    if global_conf.debug:
        breakpoint()
    return dataset
