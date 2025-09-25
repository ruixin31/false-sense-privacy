import re
import itertools
import datasets
from rouge_score import rouge_scorer
# from presidio_analyzer import AnalyzerEngine
import numpy as np
from statistics import mode
from tqdm import tqdm

from src.utils import (
    PromptWithCache,
    wrap_dataset_map_func,
    get_access_key,
    proper_slice_array_by_range,
)

import nltk

# analyzer = AnalyzerEngine()
pii_datasets_medqa = "exps/scrub_msft-medqa_factorized-llama-1000/scrub_msft-medqa_factorized-llama-1000/dataset"
pii_datasets_wildchat = "exps/scrub_msft-wildchat_all_unprocessed-llama-1000/scrub_msft-wildchat_all_unprocessed-llama-1000/dataset"
processed_dataset = {}
def process_dataset(dataset_name):
    if dataset_name in processed_dataset:
        return processed_dataset[dataset_name]

    if 'medqa' in dataset_name:
        dataset_path = pii_datasets_medqa
    elif 'wildchat' in dataset_name:
        dataset_path = pii_datasets_wildchat
    else:
        raise ValueError("Invalid dataset")
    dataset = datasets.load_from_disk(dataset_path)
    processed_dataset[dataset_name] = dataset['entities']
    return dataset['entities']


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
    dataset_name,
    **kwargs,
):
    result = {}
    # for entry in tqdm(dataset):
    exp_id = "pii_existence"
    labels = [
        f"{exp_id}_{scores}"
        # Assert there is only one category
        for scores in [
            "precision_score",
            "recall_score",
            "recall_score_exclude_input",
        ]
    ]
    if dataset_name == "medqa":
        entities_dataset = process_dataset(dataset_name)
    else:
        entities_dataset = process_dataset(dataset_name)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    for range_txt, single_range in zip(range_maps, ranges):
        san_fact_on_orig_context = {
            "score": [],
            "missing": [],
        }
        orig_fact_on_san_context = {
            "score": [],
            "missing": [],
        }
        orig_fact_exclude_input_on_san_context = {
            "score": [],
            "missing": [],
        }
        for entry in tqdm(dataset):
            # top_article_ids = entry["article_idx"]
            # top_article_ids = top_article_ids[single_range]
            # if not top_article_ids:
            #     # If we don't have a valid top article, we skip this entry
            #     continue
            # top_article_idx = mode(list(map(lambda x: x[0], top_article_ids)))
            top_article_idx = get_top_article_idx(entry, single_range)
            if top_article_idx is None:
                continue
            this_article_id = entry["id"]
            context_orig = entry[context_key_orig]
            facts_orig = entry[facts_key_orig]
            context_sanitized = dataset[top_article_idx][context_key_sanitized]
            facts_sanitized = dataset[top_article_idx][facts_key_sanitized]

            scores = scorer.score(context_orig, context_sanitized)["rougeL"]

            precision_score = scores.precision
            recall_score = scores.recall

            # san_fact_on_orig_context["score"].append(precision_score)
            # orig_fact_on_san_context["score"].append(recall_score)

            # Remove the facts used to match
            sentences = [
                sentence
                for line in context_orig.split("\n")
                for sentence in nltk.sent_tokenize(line)
            ]
            aux_range = cfc_range if cfc_range else single_range
            orig_facts_used_for_matching = proper_slice_array_by_range(
                aux_range, entry, facts_orig
            )
            # orig_facts_used_for_matching = slice_array_by_range(
            #     single_range, facts_orig
            # )
            idx_to_remove = set()
            for fact in orig_facts_used_for_matching:
                rouge_scores = [
                    # Choosing recall as there might be more information in the original context
                    scorer.score(fact, sentence)["rougeL"].recall
                    for sentence in sentences
                ]
                idx_to_remove.add(np.argmax(rouge_scores))

            fixed_context_orig = " ".join(
                [sentences[i] for i in range(len(sentences)) if i not in idx_to_remove]
            )

            entities = entities_dataset[top_article_idx]

            sentences_len = list(map(len, sentences))
            starting_idx = list(itertools.accumulate([0] + sentences_len[:-1]))

            possible_entities = []
            for entity in entities:
                entity_start = entity["offset"]
                entity_len = entity["length"]
                entity_end = entity_start + entity_len - 1
                # entity_text = entity["text"]
                # entity_type = entity["type"]
                starting_sentence_idx = -1
                ending_sentence_idx = -1
                for i, (start, l) in enumerate(zip(starting_idx, sentences_len)):
                    if entity_start >= start and starting_sentence_idx == -1:
                        starting_sentence_idx = i
                    if entity_end < start + l:
                        ending_sentence_idx = i
                        break
                else:
                    # Entity not found in any sentence, possibily due to truncation 
                    continue
                entity['starting_sentence_idx'] = starting_sentence_idx
                entity['ending_sentence_idx'] = ending_sentence_idx
                possible_entities.append(entity)
            del entities
            
            if not possible_entities:
                orig_fact_exclude_input_on_san_context["missing"].append(this_article_id)
                orig_fact_on_san_context["missing"].append(this_article_id)
                continue
            
            entity_existence_scores = [entity["text"].lower() in context_sanitized.lower() for entity in possible_entities]
            orig_fact_on_san_context["score"].append(sum(entity_existence_scores) / len(entity_existence_scores))
            
            cleaned_entities = []
            for entity in possible_entities:
                starting_sentence_idx = entity['starting_sentence_idx']
                ending_sentence_idx = entity['ending_sentence_idx']

                for i in range(starting_sentence_idx, ending_sentence_idx + 1):
                    if i in idx_to_remove:
                        break
                else:
                    cleaned_entities.append(entity)
            del possible_entities
            
            if not cleaned_entities or not fixed_context_orig:
                if not fixed_context_orig:
                    assert not cleaned_entities
                orig_fact_exclude_input_on_san_context["missing"].append(this_article_id)
                continue

            cleaned_entity_existence_scores = [entity["text"].lower() in context_sanitized.lower() for entity in cleaned_entities]

            
            # recall_exclude_input_scores = scorer.score(
            #     fixed_context_orig, context_sanitized
            # )["rougeL"].recall

            orig_fact_exclude_input_on_san_context["score"].append(
                sum(cleaned_entity_existence_scores) / len(cleaned_entity_existence_scores)
            )
            # orig_fact_exclude_input_on_san_context_pfive_thres["score"].append(
            #     recall_exclude_input_scores > 0.5
            # )

        # breakpoint()
        # just using the framework over here
        data = [
            str(sum(scores[cat]) / len(scores[cat])) if scores[cat] else "N/A"
            for cat in ["score"]
            for scores in [
                san_fact_on_orig_context,
                orig_fact_on_san_context,
                orig_fact_exclude_input_on_san_context,
            ]
        ]
        data_str = "\t".join(data)
        result[range_txt] = dict(zip(labels, data))

    print_and_write_results_json(result)
    print_and_write_results(f"done")
    # return dataset
