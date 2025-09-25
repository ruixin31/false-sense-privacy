from typing import DefaultDict
from tqdm import tqdm
from rouge_score import rouge_scorer
from src.utils import PromptWithCache, proper_slice_array_by_range, get_access_key
from collections import defaultdict
from statistics import mode
from pathlib import Path
import re




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
    cfc_range,
    **kwargs,
):
    # Gather possible chatgpt queries

    result = {}
    exp_id = "n_gram_leakage"

    repeats = 1
    if global_conf.model.lm_model == "llama":
        repeats = 3

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rouge3"], use_stemmer=True)
    labels = [
        f"{exp_id}-{scores}"
        for scores in [
            "aux_1_gram_leakage",
            "aux_2_gram_leakage",
            "aux_3_gram_leakage",
            # "orig_fact_on_san_context",
            # "orig_fact_exclude_input_on_san_context",
        ]
    ]
    for range_txt, single_range in zip(range_maps, ranges):
                
        san_fact_on_orig_context = {
            "score": [],
            "missing": [],
        }
        orig_fact_on_san_context_1_gram = {
            "score": [],
            "missing": [],
        }
        orig_fact_on_san_context_2_gram = {
            "score": [],
            "missing": [],
        }
        orig_fact_on_san_context_3_gram = {
            "score": [],
            "missing": [],
        }
        orig_fact_exclude_input_on_san_context = {
            "score": [],
            "missing": [],
        }
        access_func = get_access_key(dataset, "matching_cues", global_conf)
        def get_aux(entry):
            field_of_concern = entry[access_func]
            if not len(field_of_concern) == 1:
                return ''
            assert len(field_of_concern) == 1
            return field_of_concern[0]
        for entry in tqdm(dataset):
            # top_article_ids = entry["article_idx"]
            # top_article_ids = top_article_ids[single_range]
            # if not top_article_ids:
            #     # If we don't have a valid top article, we skip this entry
            #     continue
            # top_article_idx = mode(list(map(lambda x: x[0], top_article_ids)))
            # top_article_idx = get_top_article_idx(entry, single_range)
            # if top_article_idx is None:
            #     continue
            this_article_id = entry["id"]
            context_orig = entry[context_key_orig]
            facts_orig = entry[facts_key_orig]
            # context_sanitized = dataset[top_article_idx][context_key_sanitized]
            # facts_sanitized = dataset[top_article_idx][facts_key_sanitized]

            aux_range = cfc_range if cfc_range else single_range
            selected_facts = proper_slice_array_by_range(aux_range, entry, entry[facts_key_orig])

            # fused_used_facts = " ".join(selected_facts)
            fused_used_facts = get_aux(entry)
            if not fused_used_facts:
                continue
            
            # breakpoint()
            # scores = scorer.score(context_orig, fused_used_facts)
            scores = scorer.score(fused_used_facts, context_orig)
            recall_score = scores['rouge1'].recall
            orig_fact_on_san_context_1_gram["score"].append(recall_score)
            recall_score = scores['rouge2'].recall
            orig_fact_on_san_context_2_gram["score"].append(recall_score)
            recall_score = scores['rouge3'].recall
            orig_fact_on_san_context_3_gram["score"].append(recall_score)

        data = [
            str(sum(scores[cat]) / len(scores[cat])) if scores[cat] else "N/A"
            for cat in ["score"]
            for scores in [
                # san_fact_on_orig_context,
                orig_fact_on_san_context_1_gram,
                orig_fact_on_san_context_2_gram,
                orig_fact_on_san_context_3_gram,
                # orig_fact_exclude_input_on_san_context,
            ]
        ]
        result[range_txt] = dict(zip(labels, data))

    # exit()
    print_and_write_results_json(result)
    print_and_write_results(f"done")
