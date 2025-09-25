import multiprocessing
from src.utils import get_access_key
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
    # context_key_orig = get_access_key(dataset, "original_document", global_conf)
    context_key_sanitized = get_access_key(dataset, "sanitized_document", global_conf)
    facts_key_sanitized = get_access_key(dataset, "sanitized_facts", global_conf)
    facts_key_orig = get_access_key(dataset, "original_facts", global_conf)


    key = get_access_key(
        dataset, global_conf.evaluation.privacy.samples_key_from_orig, global_conf
    )
    # key_sanitized = regex_find(cfg.evaluation.privacy.samples_key_from_sanitized, dataset.column_names)
    key_sanitized = get_access_key(
        dataset, global_conf.evaluation.privacy.samples_key_from_sanitized, global_conf
    )


    if "use" in method_name_config:
        matching_method = method_name_config["use"]
    else:
        raise
    
    
    if "cfc" in matching_method:
        facts_key_orig = key

    def get_string_matching(entry):
        orig_facts = entry[facts_key_orig]
        scores_per_fact = []
        article_idx_per_fact = []
        # breakpoint()
        for orig_fact in orig_facts:
            scores = []
            article_idxes = []
            for sanitized_entry_idx, sanitized_entry in enumerate(dataset):
                context = sanitized_entry[context_key_sanitized]

                if matching_method == "context":
                    scores.append(scorer.score(orig_fact, context)["rougeL"].recall)
                    article_idxes.append(sanitized_entry_idx)

                if matching_method == "cfc_context":
                    scores.append(scorer.score(orig_fact, context)["rougeL"].recall)
                    article_idxes.append(sanitized_entry_idx)

                elif matching_method == "facts":
                    sanitized_facts = sanitized_entry[facts_key_sanitized]
                    for sanitized_fact in sanitized_facts:
                        scores.append(
                            scorer.score(orig_fact, sanitized_fact)["rougeL"].recall
                        )
                        article_idxes.append(sanitized_entry_idx)
                else:
                    # Unsupported method
                    raise
            # Sort both scores and article_idxes by scores
            paired = list(zip(scores, article_idxes))
            paired.sort(reverse=True)  # Sort in descending order

            scores, article_idxes = zip(*paired)
            scores, article_idxes = scores[:100], article_idxes[:100]
            scores_per_fact.append(scores)
            article_idx_per_fact.append(article_idxes)

        return {
            "scores": scores_per_fact,
            "article_idx": article_idx_per_fact,
        }

    num_cpus = multiprocessing.cpu_count()
    # num_cpus = 122
    dataset = dataset.map(get_string_matching, num_proc=num_cpus)
    # dataset = dataset.map(get_string_matching)

    return dataset
