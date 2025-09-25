import json
import random


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
    # Privacy-specific arguments
    facts_key_orig,
    facts_key_sanitized,
    context_key_orig,
    context_key_sanitized,
    range_maps,
    ranges,
    get_top_article_idx,
    **kwargs,
):
    result = {}

    for range_txt, single_range in zip(range_maps, ranges):
        empty_matching = 0

        for entry in dataset:
            top_article_idx = get_top_article_idx(entry, single_range)

            if top_article_idx is None:
                empty_matching += 1

        result[range_txt] = empty_matching / len(dataset)

    print_and_write_results_json(result)
    print_and_write_results(f"done")
