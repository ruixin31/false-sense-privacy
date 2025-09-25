import logging
from pathlib import Path
import json
import pickle
import time
from tqdm import tqdm
import pdb
from omegaconf import ListConfig
from omegaconf.omegaconf import OmegaConf, open_dict


import numpy as np
import torch
import transformers

import contriever.src.index
import contriever.src.contriever
import contriever.src.utils
import contriever.src.slurm
from contriever.src.evaluation import calculate_matches
import contriever.src.normalize_text

# from src.data import load_eval_data
from src.index import (
    Indexer,
    get_index_dir_and_passage_paths,
    BM25Index,
    get_bm25_index_data_dir,
    get_index_passages_and_id_map,
)
from src.utils import get_access_key

# from src.gpt import OpenAISynthesis

from datasets import load_from_disk, load_dataset
from pprint import pprint

from load_dataset import load_dataset_from_config

from gritlm import GritLM


def embed_queries(args, queries, model, tokenizer):
    model.eval()
    embeddings, batch_question = [], []
    with torch.no_grad():

        for k, q in enumerate(tqdm(queries, desc="Embedding queries")):
            if not type(model) == GritLM:
                if args.lowercase:
                    q = q.lower()
                if args.normalize_text:
                    q = contriever.src.normalize_text.normalize(q)
            batch_question.append(q)

            if len(batch_question) == args.per_gpu_batch_size or k == len(queries) - 1:

                if type(model) == GritLM:
                    output = torch.tensor(model.encode(batch_question))

                else:
                    encoded_batch = tokenizer.batch_encode_plus(
                        batch_question,
                        return_tensors="pt",
                        max_length=args.question_maxlength,
                        padding=True,
                        truncation=True,
                    )
                    encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                    output = model(**encoded_batch)
                embeddings.append(output.cpu())

                batch_question = []

    if len(embeddings) == 0:
        breakpoint()
    embeddings = torch.cat(embeddings, dim=0)
    # print(f"Questions embeddings shape: {embeddings.size()}")

    return embeddings.numpy()


import re


def post_process_facts_1(entry):
    # Regular expression to match curly braces along with their contents
    # Find all matches of the pattern in the text
    pattern = re.compile(r"claim ?.?: ?(.+?)\n", re.IGNORECASE)
    try:
        matches = re.findall(pattern, entry["responses"])
        subclaims = matches
        # if len(subclaims) == 0:
        #     breakpoint()
        #     raise
        # subclaims = [match[0] for match in matches]
        # subclaims = json.loads(subclaims)
    except Exception as ex:
        print(ex)
        print("Failed to parse as jsonl")
        print(entry["responses"])
        subclaims = None
    return {"subclaims": subclaims}


def privacy_eval(
    conf,
    global_conf,
    dataset,
    access_func,
):
    cfg = global_conf

    # dataset, access_func = load_dataset_from_config(cfg.datastore.dataset_in)

    if "grit" in cfg.datastore.embedding.model_name_or_path.lower():
        model = GritLM(cfg.datastore.embedding.model_name_or_path, torch_dtype="auto", mode="embedding")
        tokenizer = None
    elif "bm25" in cfg.datastore.embedding.model_name_or_path.lower():
        model = 'bm25'
        tokenizer = None

    else:
        model, tokenizer, _ = contriever.src.contriever.load_retriever(
            cfg.datastore.embedding.model_name_or_path
        )
        model.eval()
        model = model.cuda()

    eval_args = cfg.evaluation
    index_args = cfg.datastore.index

    if model == "bm25":
        index_dir, data_dir = get_bm25_index_data_dir(cfg)
        searcher = BM25Index(index_dir, data_dir, None)
    else:
        index_dir, _ = get_index_dir_and_passage_paths(cfg)
        index = Indexer(
            index_args.projection_size, index_args.n_subquantizers, index_args.n_bits, index_args.metric
        )
        index.deserialize_from(index_dir)

    embedding_shard_mapping_save_path = (
        cfg.datastore.embedding.embedding_dir + f"/passage_mapping.pkl"
    )
    with open(embedding_shard_mapping_save_path, mode="rb") as file:
        mapping = pickle.load(file)

    # assert len(dataset) == index.index.ntotal, f"number of documents
    # {len(dataset)} and number of embeddings {index.index.ntotal} mismatch"

    # breakpoint()
    # for idx, entry in tqdm(enumerate(dataset)):
    # key = cfg.evaluation.privacy.samples_key_from_orig
    key = get_access_key(
        dataset, cfg.evaluation.privacy.samples_key_from_orig, global_conf
    )
    # key_sanitized = regex_find(cfg.evaluation.privacy.samples_key_from_sanitized, dataset.column_names)
    key_sanitized = get_access_key(
        dataset, cfg.evaluation.privacy.samples_key_from_sanitized, global_conf
    )
    
    print(f"Privacy eval using access function key: {key} and key_sanitized: {key_sanitized}")

    def process(entries, idxs):
        flattened_queries = []
        for queries, idx in zip(entries[key], idxs):
            flattened_queries.extend(queries)
        if model == "bm25":
            top_ids_and_scores = []
            for query in tqdm(flattened_queries):
                result = searcher.search(query, 1)
                if len(result):
                    doc = json.loads(result[0])
                    top_ids_and_scores.append(([doc["id"]], [1]))
                else:
                    top_ids_and_scores.append(([], []))
        else:
            questions_embedding = embed_queries(eval_args.search, flattened_queries, model, tokenizer)

            top_ids_and_scores = index.search_knn(
                questions_embedding, eval_args.search.n_docs
            )
            del questions_embedding
        top_ids_and_scores_it = iter(top_ids_and_scores)
        del top_ids_and_scores
        claim_ranks_batch = []
        article_idxes_batch = []
        scores_batch = []
        top_matched_facts_batch = []

        for queries, idx in zip(entries[key], idxs):
            # breakpoint()
            # queries = access_func(entry)
            # gpt_method = OpenAISynthesis(dataset)
            # queries = facts_pipeline(question, gpt_method)
            # queries = dataset_complete['subclaims'][idx]
            # pprint(f'target question: {question}')
            # pprint(f'subclaims: {queries}')
            # breakpoint()

            # claim_ranks_batch.append([])
            # article_idxes_batch.append([])
            # top_matched_facts_batch.append([])
            # return {"claim_ranks": [], "article_idx": [], "matched_facts": []}

            # logging.info(f"Search time: {time.time()-start_time_retrieval:.1f} s.")

            claim_ranks = []
            article_idxes = []
            scores = []
            top_matched_facts = []
            claim_ranks_batch.append(claim_ranks)
            article_idxes_batch.append(article_idxes)
            scores_batch.append(scores)
            top_matched_facts_batch.append(top_matched_facts)
            if len(queries) == 0:
                continue
            for query in queries:
                (doc_ids, score) = next(top_ids_and_scores_it)
                # print(f'For fact: {query}')
                translated_idxs = list(map(lambda x: mapping[int(x)], doc_ids))
                # breakpoint()
                if len(translated_idxs):
                    top_matched_fact_idx = translated_idxs[0]
                    # try:
                    top_matched_fact = dataset[top_matched_fact_idx[0]][key_sanitized][
                        top_matched_fact_idx[1]
                    ]
                    # except:
                    #     breakpoint()
                    article_idx, _ = list(zip(*translated_idxs))
                else:
                    top_matched_fact = None
                    article_idx = []
                top_matched_facts.append(top_matched_fact)
                rank_of_truth = -1
                try:
                    rank_of_truth = article_idx.index(idx)
                except:
                    # breakpoint()
                    pass
                claim_ranks.append(rank_of_truth)
                article_idxes.append(article_idx)
                scores.append(score)

                # for (record_id, fact_id), score in zip(translated_idxs[:3], score):
                #     print(f'record_id: {record_id}, fact: {dataset["subclaims"][record_id][fact_id]}, score: {score}')
                # # print(f'score: {score}')

        return {
            "claim_ranks": claim_ranks_batch,
            "article_idx": article_idxes_batch,
            "scores": scores_batch,
            "matched_facts": top_matched_facts_batch,
        }

    # breakpoint()
    # dataset = dataset.map(lambda x: post_process_facts_1(x)) # Doing this to avoid function hashing
    dataset = dataset.map(process, with_indices=True, batched=True, batch_size=2048)
    return dataset
    # dataset.save_to_disk(cfg.evaluation.privacy.rank_dir)
    # breakpoint()
    # df = dataset.to_pandas()
    # df.to_csv(Path(cfg.evaluation.privacy.rank_dir) / "data.csv", index=False)
