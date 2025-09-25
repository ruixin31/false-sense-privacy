import re
import numpy as np
from statistics import mode
from tqdm import tqdm

from src.utils import PromptWithCache, wrap_dataset_map_func, get_access_key, proper_slice_array_by_range

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed_passages(batch_text):
    flattened_batch_text = [item for sublist in batch_text for item in sublist]
    encoded_flattened_batch_text = model.encode(flattened_batch_text)
    length = list(map(len, batch_text))
    all_length = np.cumsum(length)
    encoded_batch_text = [encoded_flattened_batch_text[i:j, :] for i,j in zip([0] + all_length.tolist()[:-1], all_length)]

    return encoded_batch_text
    # encoded_batch = tokenizer.batch_encode_plus(
    #     batch_text,
    #     return_tensors="pt",
    #     max_length=args.passage_maxlength,
    #     padding=True,
    #     truncation=True,
    # )
    #
    # encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
    # embeddings = model(**encoded_batch)  # shape: (per_gpu_batch_size, hidden_size)
    #
    # embeddings = embeddings.cpu()

    # return embeddings



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
    original_facts_key = get_access_key(dataset, "original_facts", global_conf)
    embed_original_facts_key = f"{original_facts_key}_embed"
    dataset = dataset.map(
        wrap_dataset_map_func(
            embed_passages, original_facts_key, embed_original_facts_key
        ),
        batched=True,
    )

    sanitized_facts_key = get_access_key(dataset, "sanitized_facts", global_conf)
    embed_sanitized_facts_key = f"{sanitized_facts_key}_embed"
    dataset = dataset.map(
        wrap_dataset_map_func(
            embed_passages, sanitized_facts_key, embed_sanitized_facts_key
        ),
        batched=True,
    )

    result = {}
    # for entry in tqdm(dataset):
    exp_id = 'embed'
    # this assume that there is only one category
    labels = [
        f"{exp_id}_{scores}"
        # Assert there is only one category
        for scores in [
            "san_fact_on_orig_context",
            "orig_fact_on_san_context",
            "orig_fact_exclude_input_on_san_context",
        ]
    ]

    for range_txt, single_range in zip(range_maps, tqdm(ranges)):
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
        for entry in dataset:
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
            embed_orig = np.array(entry[embed_original_facts_key])
            context_sanitized = dataset[top_article_idx][context_key_sanitized]
            facts_sanitized = dataset[top_article_idx][facts_key_sanitized]
            embed_sanitized = np.array(dataset[top_article_idx][embed_sanitized_facts_key])
            
            if embed_sanitized.size == 0:
                continue

            dot_product = np.dot(embed_orig, embed_sanitized.T)
            
            precision_scores = np.max(dot_product, axis=1)
            recall_scores = np.max(dot_product, axis=0)

            r_full = list(range(len(precision_scores)))
            # single_range is only defined on the entry
            assert len(r_full) == len(facts_orig)

            # r_selected = r_full[single_range]
            aux_range = cfc_range if cfc_range else single_range
            r_selected = proper_slice_array_by_range(aux_range, entry, r_full)
            r_complement = list(set(r_full) - set(r_selected))
            precision_exclude_input_scores = [
                precision_scores[idx] for idx in r_complement
            ]

            if len(precision_scores) > 0:
                san_fact_on_orig_context["score"].append(np.mean(precision_scores))
            if len(recall_scores) > 0:
                orig_fact_on_san_context["score"].append(np.mean(recall_scores))

            if len(precision_exclude_input_scores) > 0:
                orig_fact_exclude_input_on_san_context["score"].append(np.mean(precision_exclude_input_scores))

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
