import re
from rouge_score import rouge_scorer
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
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


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
    result = {}
    # for entry in tqdm(dataset):
    exp_id = "matched_rouge"
    labels = [
        f"{exp_id}_{scores}"
        # Assert there is only one category
        for scores in [
            "precision_score",
            "fmeasure_score",
            "recall_score",
            "recall_score_exclude_input",
            "recall_score_exclude_input_pfive_thres",
            "fmeasure_score_pfive_thres",
            "recall_score_exclude_input_wo_stop",
            "recall_score_wo_stop",
        ]
    ]

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    stop_words = set(stopwords.words('english'))

    for range_txt, single_range in zip(range_maps, ranges):
        san_fact_on_orig_context = {
            "score": [],
            "missing": [],
        }
        og_rouge_score = {
            "score": [],
            "missing": [],
        }
        og_rouge_score_pfive_thres = {
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
        orig_fact_exclude_input_on_san_context_pfive_thres = {
            "score": [],
            "missing": [],
        }
        orig_fact_exclude_input_on_san_context_stopwords_removed = {
            "score": [],
            "missing": [],
        }
        orig_fact_on_san_context_stopwords_removed = {
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
            fmeasure_score = scores.fmeasure
            recall_score = scores.recall

            san_fact_on_orig_context["score"].append(precision_score)
            og_rouge_score["score"].append(fmeasure_score)
            orig_fact_on_san_context["score"].append(recall_score)
            og_rouge_score_pfive_thres["score"].append(fmeasure_score > 0.5)

            word_tokens = word_tokenize(context_orig)
            filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
            filtered_sentence_orig = ' '.join(filtered_sentence)

            # word_tokens = word_tokenize(context_sanitized)
            # filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
            # filtered_sentence_sanitized = ' '.join(filtered_sentence)
            # print("orig_scores" + str(scores))
            scores = scorer.score(
                # filtered_sentence_orig, filtered_sentence_sanitized
                filtered_sentence_orig, context_sanitized
            )["rougeL"]
            recall_score_stopwords_removed = scores.recall
            # print("stopwords_removed scores" + str(scores))
            orig_fact_on_san_context_stopwords_removed["score"].append(recall_score_stopwords_removed)
            # breakpoint()


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
                if rouge_scores:
                    idx_to_remove.add(np.argmax(rouge_scores))

            fixed_context_orig = " ".join(
                [sentences[i] for i in range(len(sentences)) if i not in idx_to_remove]
            )
            # if not fixed_context_orig:
            #     orig_fact_exclude_input_on_san_context["missing"].append(this_article_id)
            #     orig_fact_exclude_input_on_san_context_pfive_thres["missing"].append(
            #         this_article_id
            #     )
            #     continue
            recall_exclude_input_scores = scorer.score(
                fixed_context_orig, context_sanitized
            )["rougeL"].recall
            
            word_tokens = word_tokenize(fixed_context_orig)
            filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
            filtered_sentence = ' '.join(filtered_sentence)
            recall_exclude_input_scores_stopwords_removed = scorer.score(
                filtered_sentence, context_sanitized
            )["rougeL"].recall

            orig_fact_exclude_input_on_san_context["score"].append(
                recall_exclude_input_scores
            )
            orig_fact_exclude_input_on_san_context_stopwords_removed["score"].append(
                recall_exclude_input_scores_stopwords_removed
            )
            orig_fact_exclude_input_on_san_context_pfive_thres["score"].append(
                recall_exclude_input_scores > 0.5
            )

        # breakpoint()
        # just using the framework over here
        data = [
            str(sum(scores[cat]) / len(scores[cat])) if scores[cat] else "N/A"
            for cat in ["score"]
            for scores in [
                san_fact_on_orig_context,
                og_rouge_score,
                orig_fact_on_san_context,
                orig_fact_exclude_input_on_san_context,
                orig_fact_exclude_input_on_san_context_pfive_thres,
                og_rouge_score_pfive_thres,
                orig_fact_exclude_input_on_san_context_stopwords_removed,
                orig_fact_on_san_context_stopwords_removed,
            ]
        ]
        data_str = "\t".join(data)
        result[range_txt] = dict(zip(labels, data))

    print_and_write_results_json(result)
    print_and_write_results(f"done")
    # return dataset
