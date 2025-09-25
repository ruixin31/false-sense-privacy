# from https://github.com/RulinShao/Scaling/blob/e508faf0b7f8c272c6856b1c184b34d88728929e/src/embed.py
import os

import argparse
import csv
import logging
import pickle
import pdb
from tqdm import tqdm
import json

from datasets import load_dataset, load_from_disk
from load_dataset import load_dataset_from_config

import numpy as np
import torch

import transformers

import contriever.src.contriever
import contriever.src.utils
import contriever.src.normalize_text

from gritlm import GritLM

from src.index import get_bm25_index_data_dir


def embed_passages(args, passages, model, tokenizer):
    total = 0
    allids, allembeddings = [], []
    batch_ids, batch_text = [], []
    with torch.no_grad():
        for k, p in tqdm(enumerate(passages)):
            batch_ids.append(p["id"])
            if args.no_title or not "title" in p:
                text = p["text"]
            else:
                text = p["title"] + " " + p["text"]
            if not type(model) == GritLM:
                if args.lowercase:
                    text = text.lower()
                if args.normalize_text:
                    text = contriever.src.normalize_text.normalize(text)
            batch_text.append(text)

            if len(batch_text) == args.per_gpu_batch_size or k == len(passages) - 1:
                if type(model) == GritLM:
                    embeddings = torch.tensor(model.encode(batch_text, batch_size=args.per_gpu_batch_size))

                else:
                    encoded_batch = tokenizer.batch_encode_plus(
                        batch_text,
                        return_tensors="pt",
                        max_length=args.passage_maxlength,
                        padding=True,
                        truncation=True,
                    )

                    encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                    embeddings = model(
                        **encoded_batch
                    )  # shape: (per_gpu_batch_size, hidden_size)
                    embeddings = embeddings.cpu()

                total += len(batch_ids)
                allids.extend(batch_ids)
                allembeddings.append(embeddings)

                batch_text = []
                batch_ids = []
                if k % 10000 == 0 and k > 0:
                    print(f"Encoded passages {total}")

    allembeddings = torch.cat(allembeddings, dim=0).numpy()
    return allids, allembeddings


def generate_passage_embeddings(transform_conf, all_conf):
    # todo: support distributed mode
    # contriever.src.slurm.init_distributed_mode(args)

    conf = all_conf.datastore.embedding
    logging.info(f"Loading retriever model from {conf.model_name_or_path}...")
    if "grit" in conf.model_name_or_path.lower():
        model = GritLM(conf.model_name_or_path, torch_dtype="auto", mode="embedding")
        tokenizer = None
    elif "bm25" in conf.model_name_or_path.lower():
        model = 'bm25'
        tokenizer = None
    else:
        model, tokenizer, _ = contriever.src.contriever.load_retriever(
            conf.model_name_or_path
        )
        model.eval()
        model = model.cuda()
        if not conf.no_fp16:
            model = model.half()

    # dataset = load_from_disk(all_conf.datastore.prepare_dataset.product_path)
    dataset, access_func = load_dataset_from_config(all_conf.datastore.dataset_in)
    
    print(f"Building embeddings using access function {access_func}")

    # flatten atomic facts and create a mapping from fact id to index
    # should abstract into a function and make it more efficient
    mapping = {}
    passages = []
    # if all_conf.transform_dataset.method == 'facts' or all_conf.transform_dataset.method == 'identity':
    # if 'facts' in all_conf.transform_dataset.exp_name:

    # Assume that this is after the facts pipeline
    if "facts" in transform_conf.exp_name:
        id = 0
        record_id = 0
        for record in dataset:
            claim_in_record_id = 0
            for subclaim in access_func(record):
                mapping[id] = (record_id, claim_in_record_id)
                if model == 'bm25':
                    passages.append({"contents": subclaim, "id": id})
                else:
                    passages.append({"text": subclaim, "id": id})

                id += 1
                claim_in_record_id += 1
            record_id += 1

        assert id == len(passages)
    else:
        id = 0
        record_id = 0
        for record in dataset:
            claim_in_record_id = 0
            text = access_func(record)
            assert type(text) == str
            mapping[id] = (record_id, claim_in_record_id)
            if model == 'bm25':
                passages.append({"contents": text, "id": id})
            else:
                passages.append({"text": text, "id": id})

            id += 1
            record_id += 1

        assert id == len(passages)

    print("Flattened and established mapping from id to index.")

    embedding_shard_save_path = conf.embedding_dir + f"/passage_embeddings.pkl"
    embedding_shard_mapping_save_path = conf.embedding_dir + f"/passage_mapping.pkl"

    if os.path.exists(embedding_shard_save_path) and conf.get(
        "use_saved_if_exists", "true"
    ):
        print(f"Embeddings exist in {embedding_shard_save_path}")
        return

    # shard_passages = fast_load_jsonl_shard(args, shard_id)

    # todo: improve the efficiency of embedding in sumsampling
    # breakpoint()

    if model == 'bm25':
        index_dir, data_dir = get_bm25_index_data_dir(all_conf)
        # write the passages to a jsonl file
        with open(data_dir + "/passages.jsonl", "w") as file:
            for passage in passages:
                file.write(json.dumps(passage) + "\n")
    else:
        allids, allembeddings = embed_passages(conf, passages, model, tokenizer)

    os.makedirs(conf.embedding_dir, exist_ok=True)
    if model != 'bm25':
        print(f"Saving {len(allids)} passage embeddings to {embedding_shard_save_path}.")
        with open(embedding_shard_save_path, mode="wb") as file:
            pickle.dump((allids, allembeddings), file)

    with open(embedding_shard_mapping_save_path, mode="wb") as file:
        pickle.dump(mapping, file)

    # print(
    #     f"Processed {len(allids)} passages in the {0}-th (out of {1}) shard.\nWritten to {embedding_shard_save_path}."
    # )
