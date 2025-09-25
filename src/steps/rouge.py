from rouge_score import rouge_scorer


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

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    rouge_l = []
    # # For debugging
    # breakpoint()
    # sanitized = access_func(dataset[0])
    # for entry in dataset_1:
    #     context = entry["context"]
    #     rouge_l.append(scorer.score(context, sanitized)["rougeL"].recall)
    for entry in dataset:
        context = entry["context"]
        sanitized = access_func(entry)
        rouge_l.append(scorer.score(context, sanitized)["rougeL"].recall)
    print_and_write_results(f"Rouge-L: {sum(rouge_l) / len(rouge_l)}")

    print_and_write_results_json(
        {
            "Rouge-L": sum(rouge_l) / len(rouge_l),
        }
    )
    return
