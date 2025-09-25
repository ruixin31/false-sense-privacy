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
    if global_conf.debug:
        import nltk, re
        def replace_consecutive_stars(input_string):
            # Replace all consecutive asterisks with a single asterisk
            return re.sub(r'\*+', '*', input_string)
        tot = 0
        tot_char = 0
        empty = 0
        for entry in dataset:
            text_orig = entry[context_key_sanitized]
            text = replace_consecutive_stars(entry[context_key_sanitized])
            text = nltk.word_tokenize(text)
            if len(text) == 0:
                empty += 1
                continue
            tot += text.count('*') / len(text)
            tot_char += text_orig.count('*') / len(text_orig)
            
        print(tot / (len(dataset) - empty))
        print(tot_char / (len(dataset) - empty))
        breakpoint()




    if global_conf.debug:
        with open(
            "playground/paraphrase_matching_debug/incorrect_matching.json", "w"
        ) as f:
            pass
        incorrect_matchings = []

    for range_txt, single_range in zip(range_maps, ranges):
        correct_matching = 0
        if global_conf.debug:
            breakpoint()

        # if global_conf.debug:
        #     single_range = 'rand3'
        #     for entry in dataset:
        #         top_article_id = get_top_article_idx(entry, single_range)
        # if global_conf.debug:
        #     chosen_idx = {}
        #     for entry in dataset:
        #         orig_fact = entry[facts_key_orig]
        #         indicies = list(range(len(orig_fact)))
        #         random.shuffle(indicies)
        #         selected_fact_indices = indicies[:3]
        #         selected_fact_indices = sorted(selected_fact_indices)
        #         chosen_idx[entry["id"]] = selected_fact_indices
        #     breakpoint()
        #     with open('playground/chosen_idxes/wildchat.json', 'w') as f:
        #     # with open('playground/chosen_idxes/medqa.json', 'w') as f:
        #         json.dump(chosen_idx, f, indent=4)
                



        for entry in dataset:
            # top_article_idx = get_top_article_idx(entry, single_range, threshold=0.9)
            # top_article_idx = get_top_article_idx(entry, single_range, threshold=None, top_articles=5)
            top_article_idx = get_top_article_idx(entry, single_range)
            # top_article_idx = get_top_article_idx(entry, single_range, threshold=0.85)
            if global_conf.debug:
                top_article_idx = get_top_article_idx(entry, single_range)
                # top_article_idx = get_top_article_idx(entry, single_range, threshold=0.95)
            if top_article_idx is None:
                continue
            top_article_id = dataset[top_article_idx]["id"]
            this_article_id = entry["id"]

            correct_matching += top_article_id == this_article_id
            # if global_conf.debug and not top_article_id == this_article_id:
            #     breakpoint()
            if (
                global_conf.debug
                and not top_article_id == this_article_id
                and range_txt == "-3:"
            ):
                incorrect_matchings.append(entry)

        result[range_txt] = correct_matching / len(dataset)

    if global_conf.debug:
        random.shuffle(incorrect_matchings)
        incorrect_matchings = incorrect_matchings[:20]
        with open("playground/paraphrase_matching_debug/incorrect_matching.json", "a") as f:
            json.dump(incorrect_matchings, f, indent=4)
        breakpoint()

    print_and_write_results_json(result)
    print_and_write_results(f"done")
