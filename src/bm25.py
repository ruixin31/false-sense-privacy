import time
import re
import os
import pickle as pkl
import shutil
import argparse
import json
import subprocess
from tqdm import tqdm
import numpy as np

from transformers import GPTNeoXTokenizerFast
from pyserini.search.lucene import LuceneSearcher

# from src.subsample import subsample_from_dir
# from utils.timing import time_exec, Logger

class BM25Index(object):

    def __init__(self, index_dir, data_dir, stopwords):

        if not os.path.exists(index_dir):
            print ("Start building index for %s at %s" % (data_dir, index_dir))
            
            if stopwords is not None:
                command = """python -m pyserini.index.lucene \
                --collection JsonCollection \
                --input '%s' \
                --index '%s' \
                --generator DefaultLuceneDocumentGenerator \
                --storeRaw --threads 1 \
                --stopwords '%s' """ % (data_dir, index_dir, stopwords)
            else:
                command = """python -m pyserini.index.lucene \
                --collection JsonCollection \
                --input '%s' \
                --index '%s' \
                --generator DefaultLuceneDocumentGenerator \
                --storeRaw --threads 1""" % (data_dir, index_dir)

            ret_code = subprocess.run([command],
                                    shell=True,
                                    #stdout=subprocess.DEVNULL,
                                    #stderr=subprocess.STDOUT
                                    )
            if ret_code.returncode != 0:
                print("Failed to build the index")
                exit()
            else:
                print("Successfully built the index")

        self.searcher = LuceneSearcher(index_dir)

    def search(self, query, k, continuation=False, shift=False):
        hits = self.searcher.search(query, k=k)
        docs = []
        for hit in hits:
            docid = hit.docid

            if shift:
                docid = str(int(hit.docid)+1)

            raw = self.searcher.doc(docid).raw()
            input_ids = json.loads(raw)["input_ids"]

            if continuation:
                next_item = self.searcher.doc(str(int(hit.docid)+1))
                if next_item is not None:
                    next_raw = next_item.raw()
                    input_ids += json.loads(raw)["input_ids"]
                else:
                    print ("The last block retrieved, so skipping continuation...")

            docs.append(input_ids)
        return docs


def batch_merged(flatten_input_ids, max_seq_length, stride, pad_token, flatten_masks=None):
    all_input_ids = []
    all_targets = []
    prev_end_loc = 0

    for begin_loc in range(0, len(flatten_input_ids)-1, stride):
        end_loc = min(begin_loc + max_seq_length, len(flatten_input_ids)-1)
        trg_len = end_loc - prev_end_loc

        # we feed begin_loc ~ prev_end_log ~ end_log
        # but calculcate loss only for prev_end_log ~ end_log
        input_ids = flatten_input_ids[begin_loc:end_loc].copy()
        target_ids = flatten_input_ids[begin_loc+1:end_loc+1].copy()

        if flatten_masks is not None:
            for i, m in enumerate(flatten_masks[begin_loc+1:end_loc+1]):
                if not m:
                    target_ids[i] = pad_token

        target_ids[:-trg_len] = pad_token
        assert input_ids.shape==target_ids.shape

        if end_loc == len(flatten_input_ids)-1 and len(input_ids)==len(target_ids)<max_seq_length:
            pads = np.array([pad_token for _ in range(max_seq_length-len(input_ids))])
            input_ids = np.concatenate([input_ids, pads])
            target_ids = np.concatenate([target_ids, pads])

        assert len(input_ids)==len(target_ids)==max_seq_length, (begin_loc, end_loc, len(flatten_input_ids))

        all_input_ids.append(input_ids)
        all_targets.append(target_ids)

        prev_end_loc = end_loc

        if end_loc == len(flatten_input_ids)-1:
            break

    assert np.all([len(input_ids)==max_seq_length for input_ids in all_input_ids])
    assert np.all([len(input_ids)==max_seq_length for input_ids in all_targets])
    return np.stack(all_input_ids), np.stack(all_targets)

def batch(input_ids, max_seq_length, stride, pad_token):
    all_input_ids, all_targets = [], []
    for _input_ids in input_ids:
        _all_input_ids, _all_targets = batch_merged(_input_ids, max_seq_length, stride, pad_token)
        all_input_ids.append(_all_input_ids)
        all_targets.append(_all_targets)
    return np.concatenate(all_input_ids, 0), np.concatenate(all_targets, 0)
    

# @time_exec
def chunking(tokenizer, data_paths, tokenized_path, bm25_data_path):
    """
    Inputs:
    - tokenizer: tokenize the data for fixed size chunking.
    - data_paths: a list of jsonl files of the raw text.
    """
    # Tokenize the raw text
    data = []
    log_file = tokenized_path.replace('_tokenized.pkl', '_log.txt')
    
    print(f"Chunking: tokenizing...")
    if os.path.exists(tokenized_path):
        with open(tokenized_path, "rb") as f:
            input_ids = pkl.load(f)
        print(f"Loaded from tokenized path {tokenized_path}.")
    else:
        for data_path in data_paths:
            with open(data_path, "r") as file:
                for line in file:
                    text = json.loads(line)['text']
                    data.append(text.strip())     
        input_ids = tokenizer(data)["input_ids"]
        n_tokens = np.sum([len(ids) for ids in input_ids])
        with open(tokenized_path, "wb") as f:
            pkl.dump(input_ids, f)
        log = "Saved %.1f tokens in %s\n" % (n_tokens, tokenized_path)
        with open(log_file, 'a+') as f:
            f.write(log)
        print (log)

    # Get flatten tokens
    print(f"Chunking: flattening...")
    pad_token = tokenizer.eos_token_id
    assert not os.path.exists(bm25_data_path), f"{bm25_data_path} exists"
    flatten_input_ids = np.array([_id for ids in input_ids for _id in ids])
    if args.merge:
        all_input_ids, all_targets = batch_merged(flatten_input_ids, max_seq_length=args.max_seq_length, stride=args.stride, pad_token=pad_token)
    else:
        # todo: shouldn't feed flatten_input_ids
        all_input_ids, all_targets = batch(flatten_input_ids, max_seq_length=args.max_seq_length, stride=args.stride, pad_token=pad_token)

    # Build bm25.data
    print(f"Chunking: preparing data for BM25...")
    os.mkdir(bm25_data_path)
    offset = 0
    with open(os.path.join(bm25_data_path, "data.jsonl"), "w") as f:  # consider dedup after this?
        for input_ids in tqdm(all_input_ids):
            assert len(input_ids) <= args.max_seq_length
            text = tokenizer.decode(input_ids)
            f.write(json.dumps({
                "id": str(offset),
                "contents": text,
                "input_ids": input_ids.tolist()
            })+"\n")
            offset += 1
    print (f"Saved {offset} docs in {bm25_data_path}.")
    

# @time_exec
def build_dense_index(bm25_data_path, stopwords):
    bm25_index_path = os.path.join(bm25_data_path, 'bm25_index')
    print(f'Loading/building bm25 search index from {bm25_index_path}')
    searcher = BM25Index(bm25_index_path, bm25_data_path, stopwords)

def parse_tasks(task_string):
    # Split the task string by comma and strip whitespace
    return [task.strip() for task in task_string.split(',')]

def main(args):
    data_dir = os.path.join(args.raw_jsonl_data_dir, args.domain)
    output_dir = os.path.join(args.output_dir, str(args.seed))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    subsampled_data_path = os.path.join(output_dir, f"{args.domain}_{int(args.sample_size)}.jsonl")
    tokenized_path = os.path.join(output_dir, f"{args.domain}_{int(args.sample_size)}_tokenized.pkl")
    bm25_data_path = os.path.join(output_dir, f"{args.domain}_{int(args.sample_size)}.bm25_index.data")
    # logger = Logger(args)
    time_sample, time_chunk, time_index = None, None, None
    
    # if 'subsample' in args.tasks:
    #     if args.overwrite_ok and os.path.exists(bm25_data_path):
    #         shutil.rmtree(bm25_data_path)
    #     if not args.overwrite_ok:
    #         assert not os.path.exists(subsampled_data_path), f'{subsampled_data_path} exits, please pass --overwrite_ok'
    #         assert not os.path.exists(tokenized_path), f'{tokenized_path} exits, please pass --overwrite_ok'
    #         assert not os.path.exists(bm25_data_path), f'{bm25_data_path} exits, please pass --overwrite_ok'   
        
    #     _, time_sample = subsample_from_dir(args.sample_size, data_dir, subsampled_data_path, seed=args.seed)
    #     tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
    #     _, time_chunk = chunking(
    #         tokenizer, 
    #         [subsampled_data_path], 
    #         tokenized_path,
    #         bm25_data_path,
    #     )
    
    if 'index' in args.tasks:
        _, time_index = build_dense_index(bm25_data_path, stopwords=args.stopwords)
    
    if 'evaluate' in args.tasks:
        pass
    
    # logger.log_results(time_sample=time_sample, time_chunk=time_chunk, time_index=time_index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # chunking
    parser.add_argument("--merge", action="store_true", default=True)
    parser.add_argument("--stride", default=512, type=int)
    parser.add_argument("--max_seq_length", default=1024, type=int)
    # exp
    parser.add_argument("--raw_jsonl_data_dir", default='/gscratch/zlab/rulins/data/pile-domains', help='Assuming raw data is save in <raw_jsonl_data_dir>/<domain>/')
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=1e6)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--stopwords", type=str, default=None, help='path to stopwords that will be removed when building bm25 index')
    parser.add_argument("--tasks", default='subsample,index', help='Pass tasks splited by comma')
    parser.add_argument("--overwrite_ok", action='store_true')
    # out
    parser.add_argument("--output_dir", default='/gscratch/zlab/rulins/data/pile-domains/subsampled', type=str)
    parser.add_argument("--log_file", default='log.txt')
    args = parser.parse_args()
    args.tasks = parse_tasks(args.tasks) if args.tasks else []
    print(args.tasks)

    main(args)