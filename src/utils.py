import re
from functools import partial
import fcntl
import pickle
import logging
import json
import datasets
from pathlib import Path
from enum import Enum
from statistics import mode
import random
import hashlib

#
class STEP_FLAGS(Enum):
    SKIP_CHILD = 1
    # NO_DATASET = 2


def _regex_find(pattern, items):
    targets = []
    for item in reversed(items):
        if re.match(pattern, item):
            targets.append(item)
    if targets:
        targets = sorted(targets, key=len)
        if len(targets) > 1:
            logging.warn(
                f"Multiple dataset key matches found: {targets}. Choosing the shortest one: {targets[0]}"
            )
        return targets[0]
    # raise "No match found"


class Prompt:
    RERUN_POST_PROCESS = False

    def __init__(self, user_prompt, system_prompt=None, post_process_func=None):
        self.user_prompt = user_prompt
        self.system_prompt = system_prompt
        self.post_process_func = post_process_func

    def format_entry(self, input):
        return self.user_prompt.format(input)

    def apply(
        self,
        dataset,
        gpt_method,
        input_key,
        output_key,
        *,
        is_mcq=False,
        mcq_choices=None,
        processing_args=None,
    ):
        if processing_args is None:
            processing_args = {
                "batched": True,
                "batch_size": 99999999,
            }

        # A pretty ad-hoc solution to the problem of having to pass the
        # choices, could be improved by having only mcq_choices as the argument
        if not is_mcq:
            mcq_choices = None
        # Key for prompts
        preprocess_output_key = f"_responses-{output_key}_pending"
        # Key for generated responses
        process_output_key = (
            output_key if self.post_process_func is None else f"_responses-{output_key}"
        )
        assert not (Prompt.RERUN_POST_PROCESS and self.post_process_func is None)
        if (
            Prompt.RERUN_POST_PROCESS
            and self.post_process_func is not None
            and process_output_key not in dataset.column_names
            and output_key in dataset.column_names
        ):
            dataset = dataset.rename_column(output_key, process_output_key)

        assert not (
            Prompt.RERUN_POST_PROCESS and process_output_key not in dataset.column_names
        )

        if (
            not Prompt.RERUN_POST_PROCESS
            # This is a bit extra... The assertion above has already taken care of this
            or process_output_key not in dataset.column_names
        ):
            dataset = dataset.map(
                wrap_dataset_map_func(
                    self.format_entry,
                    input_key,
                    preprocess_output_key,
                ),
                desc="Formatting prompt",
            )
            logging.info(f"Example prompt: {dataset[0][preprocess_output_key]}")
            logging.info(
                f"Example response: {gpt_method.prompt(dataset[0][preprocess_output_key], choices=mcq_choices)}"
            )
            # breakpoint()

            if is_mcq:
                prompt_func = partial(gpt_method.prompt_batch, choices=mcq_choices)
            else:
                prompt_func = gpt_method.prompt_batch

            dataset = dataset.map(
                wrap_dataset_map_func(
                    prompt_func,
                    preprocess_output_key,
                    process_output_key,
                    system_prompt=self.system_prompt,
                ),
                remove_columns=[preprocess_output_key],
                desc="Inference",
                **processing_args,
            )

        if self.post_process_func is not None:
            dataset = dataset.map(
                wrap_dataset_map_func(
                    self.post_process_func,
                    process_output_key,
                    output_key,
                )
            )

        return dataset

def hash_key(key):
    return hashlib.md5(key.encode()).hexdigest()


class PromptWithCache(Prompt):
    def __init__(self, output_folder, gpt_method, post_process_func=None, repeats=None, set_alt_job_for_inference=False, exp_id=None, bin_id='v5_queries'):
        super().__init__("", "", post_process_func)
        self.repeats = repeats
        self.output_folder = output_folder
        self.cache_folder = output_folder.parent / "cache"
        self.set_alt_job_for_inference = set_alt_job_for_inference
        self.gpt_method = gpt_method
        self.cached_queries_count = 0
        self.received_queries_count = 0
        
        # Currently, this is tied to v4 version
        if set_alt_job_for_inference:
            # self.cache_central_folder = Path('cache') / exp_id
            self.cache_central_folder = self.cache_folder
            self.cache_central_folder.mkdir(parents=True, exist_ok=True)
            # self.pkl_file_name_base = 'v4_queries'
            self.pkl_file_name_base = bin_id
            pkl_file = self.cache_central_folder / f"{self.pkl_file_name_base}-resolved.pkl"
        else:
            self.pkl_file_name_base = 'queries'
        # self.pkl_file_name_base = 'queries'
            # Old pkl file location, for backward compatibility
            new_pkl_file = self.cache_folder / f"{self.pkl_file_name_base}.pkl"
            old_pkl_file = output_folder / f"{self.pkl_file_name_base}.pkl"
            if not new_pkl_file.exists() and old_pkl_file.exists():
                pkl_file = old_pkl_file
            else:
                pkl_file = new_pkl_file
                if old_pkl_file.exists():
                    old_pkl_file.unlink()

        self.to_resolve_queries = {}
        self.state = "initialize"

        if pkl_file.exists():
            with open(pkl_file, "rb") as f:
                self.resolved_queries = pickle.load(f)
                print(f"Loaded {len(self.resolved_queries)} queries from cache")
        else:
            self.resolved_queries = {}

    def prompt(self, x):
        if self.set_alt_job_for_inference:
            x_hash = hash_key(x)
        else:
            x_hash = x
        if self.state == "receiving_requests":
            self.received_queries_count += 1
            if x_hash not in self.resolved_queries:
                # if '- Claim 1: A complete blood count is within the reference range.' in x:
                #     breakpoint()
                # if x in self.to_resolve_queries:
                #     breakpoint()
                #     return self.to_resolve_queries[x]
                self.to_resolve_queries[x] = None
            else:
                self.cached_queries_count += 1
            return None

        else:
            return self.resolved_queries[x_hash]

    def resolve(self):
        if self.set_alt_job_for_inference:
            cache = self.cache_central_folder / f"{self.pkl_file_name_base}-pending.pkl"
            
            if cache.exists():
                with open(cache, "rb") as f:
                    fcntl.flock(f, fcntl.LOCK_SH)
                    self.to_resolve_queries.update(pickle.load(f))
                    fcntl.flock(f, fcntl.LOCK_UN)
            with open(cache, "wb") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                # f.seek(0)
                pickle.dump(self.to_resolve_queries, f)
                # f.truncate()
                fcntl.flock(f, fcntl.LOCK_UN)
                print(f"Updated {len(self.to_resolve_queries)} queries to cache")
            return

        keys = list(self.to_resolve_queries.keys())
        print("Resolving", len(keys), "queries out of", self.received_queries_count, "queries")
        if self.repeats is not None:
            keys = [key for key in keys for _ in range(self.repeats)]
        ans = self.gpt_method.prompt_batch(keys)
        if self.repeats is not None:
            keys = [keys[i] for i in range(0, len(keys), self.repeats)]
            ans = [ans[i : i + self.repeats] for i in range(0, len(ans), self.repeats)]
        logging.info(f"Example prompt: {keys[0]}")
        logging.info(f"Example response: {ans[0]}")

        self.resolved_queries.update(dict(zip(keys, ans)))

        # Pickle queries
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.cache_folder.mkdir(parents=True, exist_ok=True)

        cache = self.cache_folder / f"{self.pkl_file_name_base}.pkl"
        if cache.exists():
            with open(cache, "rb") as f:
                self.resolved_queries.update(pickle.load(f))
        with open(cache, "wb") as f:
            pickle.dump(self.resolved_queries, f)

        with open(self.output_folder / "inspect.json", "w") as f:
            json.dump(list(map(lambda key: [key, self.resolved_queries.get(key)], keys[:10])), f)

    def should_keep_looping(self):
        if self.state == "initialize":
            # Need to at least look at all the queries once
            self.state = "receiving_requests"
            return True

        elif self.state == "receiving_requests":
            if len(self.to_resolve_queries) > 0:
                self.resolve()

                if self.set_alt_job_for_inference:
                    return False
            self.state = "serving_requests"
            return True
        else:
            return False

    @property
    def resolved(self):
        return self.state == "serving_requests"


def print_and_write_results_template(*result, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f:
        print(*result, file=f)
        print(*result)


def print_and_write_results_json_template(update_dict, *, steps_path, step_name):
    result_file = steps_path / "results.json"
    Path(result_file).parent.mkdir(parents=True, exist_ok=True)

    # Ensure all the data is nested under the step name and prevent things from
    # getting overwritten
    update_dict = {step_name: update_dict}

    # Open the file in read+write mode
    with open(result_file, "a+") as file:
        # Lock the file
        fcntl.flock(file, fcntl.LOCK_EX)

        try:
            # Load the existing data
            file.seek(0)
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                # If the file is empty or contains invalid JSON, start with an empty dict
                data = {}

            # Update the data with the new dictionary
            data.update(update_dict)

            # Write the updated data back to the file
            file.seek(0)
            file.truncate()
            json.dump(data, file, indent=4)

        finally:
            # Unlock the file
            fcntl.flock(file, fcntl.LOCK_UN)


# def do_inference(x, gpt_method, user_prompt, *args, **kwargs):
#     # breakpoint()
#     user_content = user_prompt.format(x)
#     # breakpoint()
#     return user_content

# Note that this is for backward compatibility. it really should be called format prompt
# output = gpt_method.prompt(user_content, *args, **kwargs)
# return output


def wrap_dataset_map_func(func, input_key, output_key, *args, **kwargs):
    if isinstance(input_key, str):
        access_func = lambda x: x[input_key]
    else:
        access_func = input_key

    def wrapped_func(entry):
        output = func(access_func(entry), *args, **kwargs)
        return {output_key: output}

    return wrapped_func


def check_contain_pending_batched_output(dataset):
    column_names = dataset.column_names
    for column_name in column_names:
        if column_name.endswith("_pending"):
            return True
    return False


def free_memory():
    import gc, torch

    gc.collect()
    torch.cuda.empty_cache()


def set_access_key(dataset, key, val, global_conf=None):
    if global_conf is not None:
        assert (
            key in global_conf.dataset_key_defaults
        ), f"Key {key} not in global_conf.dataset_key_defaults"

        # Overwriting key assignment for the current step. No the best setup,
        # but it'll do the trick
        if (
            isinstance(global_conf.current_transform_step.extra_config, dict)
            and "akey" in global_conf.current_transform_step.extra_config
        ):
            key = global_conf.current_transform_step.extra_config["akey"]

    if dataset.info.post_processed is None:
        # Hacks due to https://discuss.huggingface.co/t/how-do-i-add-custom-metadata-fields-to-datasets/43714
        dataset.info.post_processed = datasets.info.PostProcessedInfo(
            resources_checksums={}
        )

    dataset.info.post_processed.resources_checksums[key] = val

def update_access_key(dataset, val, new_val, global_conf=None):
    if dataset.info.post_processed is None:
        # Hacks due to https://discuss.huggingface.co/t/how-do-i-add-custom-metadata-fields-to-datasets/43714
        dataset.info.post_processed = datasets.info.PostProcessedInfo(
            resources_checksums={}
        )

    for key, value in dataset.info.post_processed.resources_checksums.items():
        if value == val:
            dataset.info.post_processed.resources_checksums[key] = new_val
            return

def get_access_key(dataset, key, global_conf):
    assert (
        key in global_conf.dataset_key_defaults
    ), f"Key {key} not in global_conf.dataset_key_defaults"

    if dataset.info.post_processed is None:
        val = None
    else:
        val = getattr(dataset.info.post_processed, "resources_checksums", {}).get(
            key, None
        )

    if val is None:
        val_regex = global_conf.dataset_key_defaults.get(key, None)
        val = _regex_find(val_regex, dataset.column_names)
    return val



class StepRegister:
    def __init__(self):
        self.steps = []


class Default(dict):
    def __missing__(self, key):
        return f"{{{key}}}"


class PromptTemp:
    def __init__(self, user_prompt, extensions, type=None):
        self.user_prompt = user_prompt
        self.extensions = extensions
        self.type = type

    def adapt_to_count(self, count):
        d = {}
        for idx, ext in enumerate(self.extensions):
            blurb = "\n".join([ext.format(idx=c + 1) for c in range(count)])
            d[f"ext{idx+1}"] = blurb
        return self.user_prompt.format_map(Default(d))

    def format(self, txt1_record, txt1_facts, txt2_record, txt2_facts):
        pass


def slice_array_by_range(single_range, *args):
    # Assert the length of all entries in the args are the same
    first_len = len(args[0])
    assert all(first_len == len(arg) for arg in args)
    if type(single_range) == slice:
        output = zip(*list(zip(*args))[single_range])
    elif type(single_range) == list:
        transformed_items = list(zip(*args))
        try:
            output = zip(*[transformed_items[idx] for idx in single_range])
        except:
            print(single_range)
            print(transformed_items)
            breakpoint()
    else:
        num_items = min(
            first_len,
            max(
                1,
                (
                    int(single_range[:-1]) * first_len // 100
                    if "%" in single_range
                    else int(single_range)
                ),
            ),
        )
        if first_len:
            output = zip(*random.sample(list(zip(*args)), num_items))
    output = list(output)
    if len(output) == 1:
        output = output[0]
    return output


rand3_idx = {}
def set_rand3_idx(dataset_name):
    if 'dataset' in rand3_idx and 'loaded_dataset_name' in rand3_idx and rand3_idx['loaded_dataset_name'] == dataset_name:
        return
    if 'medqa' in dataset_name:
        if 'gpt-4o' in dataset_name:
            path = 'playground/chosen_idxes/medqa_gpt-4o-facts.json'
        else:
        # path = 'playground/chosen_idxes/medqa.json'
            path = 'playground/chosen_idxes/medqa.json_sfd.json'
    elif 'wildchat' in dataset_name:
        if 'gpt-4o' in dataset_name:
            assert False
        # path = 'playground/chosen_idxes/wildchat.json'
        path = 'playground/chosen_idxes/wildchat.json_sfd.json'
    else:
        return
    print(f"Setting Rand3 idx to be {path}")
    with open(path) as f:
        rand3_idx['dataset'] = json.load(f)
        rand3_idx['loaded_dataset_name'] = dataset_name
    
def set_rand3_idx_from_idx_selection(idx_selection, global_conf):
    rand3_idx['dataset'] = idx_selection
    rand3_idx['loaded_dataset_name'] = global_conf.dataset.name
        
# This api def needs to get upated 
def _process_per_entry_range(entry, single_range):
    if single_range == "rand3":
        single_range = rand3_idx['dataset'][str(entry['id'])]
    return single_range

# This is only wrong in my current setup lol
wrong = slice(None, 1, None)
def proper_slice_array_by_range(single_range, entry, *args):
    assert single_range != wrong
    single_range = _process_per_entry_range(entry, single_range)
    return slice_array_by_range(single_range, *args)

def get_top_article_idx(entry, single_range, threshold=0.95, top_articles=None):
    top_article_ids = entry["article_idx"]
    top_article_scores = entry["scores"]
    
    single_range = _process_per_entry_range(entry, single_range)

    if not top_article_ids:
        # This is something that should result in exception if not
        # explicitly handling it
        return

    top_article_ids, top_article_scores = slice_array_by_range(
        single_range, top_article_ids, top_article_scores
    )
    # if type(single_range) == slice:
    #     top_article_ids = top_article_ids[single_range]
    #     top_article_scores = top_article_scores[single_range]
    # else:
    #     num_items = min(
    #         len(top_article_ids),
    #         max(
    #             1,
    #             (
    #                 int(single_range[:-1]) * len(top_article_ids) // 100
    #                 if "%" in single_range
    #                 else int(single_range)
    #             ),
    #         ),
    #     )
    #     if len(top_article_ids):
    #         top_article_ids, top_article_scores = zip(
    #             *random.sample(
    #                 list(zip(top_article_ids, top_article_scores)), num_items
    #             )
    #         )

    # threshold = 0.999
    if top_articles is not None:
        assert threshold is None

    articles = []
    all_similar_articles = []
    # if the score is the same, also add it to the pool for selection.
    # This is great for duplicated facts
    for idxs, scores in zip(top_article_ids, top_article_scores):
        if not idxs:
            continue

        if top_articles is not None:
            articles.extend(idxs[:top_articles])
            continue

        top_score = scores[0]
        # If the worst score is identical to the best score, then we
        # risk adding noise to matching as there might be ones that were ignored
        if scores[-1] >= top_score * threshold:
            all_similar_articles.extend(idxs)
            continue

        for idx, score in zip(idxs, scores):
            # 0.1% tolerance for numerical issues. did not check data type though
            if score < top_score * threshold:
                break
            articles.append(idx)

    # if the articles are all the same, then we can't really do
    # anything other than trying our best
    if len(articles) == 0:
        articles = all_similar_articles

    # print(articles)
    if len(articles) == 0:
        # breakpoint()
        return
    top_article_idx = mode(articles)
    # if dataset[top_article_idx]['id'] != entry['id']:
    #     breakpoint()
    return top_article_idx


def parse_create_fused_cue_range(range_of_interest_str, global_conf):
    return global_conf.create_fused_cue.str_to_range_map.get(
        range_of_interest_str, None
    )
