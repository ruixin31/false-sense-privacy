from collections import defaultdict
from functools import partial
import logging
import random
import shutil
import omegaconf
from tqdm import tqdm
from src.gpt import LM
from src.load_dataset import load_dataset_from_config
from src.utils import *
import re
from pathlib import Path
import json

import pickle

import datasets

gpt_method = None



def generate_step_conf(
    method_list, global_conf, input_dataset, exp_base_name, steps_folder
):
    pipeline = []
    step_extra_config = None
    step_modify_config = None
    for method in method_list:
        # Branch into sub-pipelines
        if type(method) == omegaconf.listconfig.ListConfig:
            pipeline.extend(
                generate_step_conf(
                    method, global_conf, input_dataset, exp_base_name, steps_folder
                )
            )
            continue

        # For meta-level flow control
        if method.startswith("_"):
            method_name_config = dict(
                filter(
                    lambda x: len(x) == 2,
                    map(lambda x: x.split("_", 1), method.split("-")),
                )
            )
            if method.startswith("_mod_key"):
                new_key = method_name_config["to"]
                input_dataset.key = new_key

            # for utilties that are defined globally
            if method.startswith("_add_next_step_custom_config"):
                step_extra_config = method_name_config

            if method.startswith("_next_step_mod_existing"):
                step_modify_config = {
                    "dataset_out": input_dataset.copy(),
                    "force_write": True,
                    "do_not_skip_this_step": True,
                }
            continue

        new_step_conf = (
            global_conf.transform_dataset.default_transform_dataset_step.copy()
        )
        new_step_conf.dataset_in = input_dataset
        new_step_conf.method = method
        new_step_conf.exp_base_name = exp_base_name
        new_step_conf.steps_folder = steps_folder
        new_step_conf.extra_config = step_extra_config
        step_extra_config = {}
        omegaconf.OmegaConf.resolve(new_step_conf)
        if step_modify_config is not None:
            for config_key in step_modify_config:
                setattr(new_step_conf, config_key, step_modify_config[config_key])
            step_modify_config = None
        input_dataset = new_step_conf.dataset_out.copy()
        exp_base_name = new_step_conf.exp_name
        pipeline.append(new_step_conf)
    return pipeline


def transform_dataset(global_conf):
    transform_dataset_conf = global_conf.transform_dataset

    assert type(transform_dataset_conf.method) == omegaconf.listconfig.ListConfig

    # omegaconf.OmegaConf.resolve(transform_dataset_conf)
    input_dataset = transform_dataset_conf.dataset_in
    exp_base_name = transform_dataset_conf.exp_base_name
    steps_folder = transform_dataset_conf.steps_folder

    pipeline = generate_step_conf(
        transform_dataset_conf.method,
        global_conf,
        input_dataset,
        exp_base_name,
        steps_folder,
    )

    # Reorder execution sequence
    for step_conf in pipeline:
        if step_conf.method in transform_dataset_conf.need_place_last:
            pipeline.remove(step_conf)
            pipeline.append(step_conf)

        # This config applies to only one run and terminates all others
        if (
            transform_dataset_conf.need_rerun_post_process
            and step_conf.method in transform_dataset_conf.need_rerun_post_process
        ):
            step_conf.dataset_in = step_conf.dataset_out
            step_conf.force_write = True
            pipeline = [step_conf]
            Prompt.RERUN_POST_PROCESS = True
            transform_dataset_conf.skip_ran_pipelines = False
            break

    if transform_dataset_conf.gpt_exps_unique_load_sanitization_only.enable:
        pipeline[0].extra_config = {
            "loadds": True,
            "configs": transform_dataset_conf.gpt_exps_unique_load_sanitization_only 
        }
        

    for step_conf in pipeline:
        if (
            Path(step_conf.dataset_out.path).parent.exists()
            and not (
                transform_dataset_conf.need_rerun
                and step_conf.method in transform_dataset_conf.need_rerun
            )
            and transform_dataset_conf.get("skip_ran_pipelines", False)
            and not step_conf.get("do_not_skip_this_step", False)
            and (
                "eval" not in step_conf.exp_name
                or (Path(step_conf.dataset_out.path).parent / "results.txt").exists()
            )
        ):
            logging.info(f"skipping {step_conf.exp_name}")
            continue

        logging.info(f"\n\n************** {step_conf.exp_name} ***********")
        # do_transform should not change any of the config, so doing it here.
        global_conf.current_transform_step = step_conf
        do_transform_dataset_step(step_conf, global_conf)


def do_transform_dataset_step(step_conf, global_conf):
    # We disable caching as we are saving datasets into a new folder every time (this increases the folder size significantly but easier to lookup when necessary)
    # It was especially annoying when developing when I don't want to cache anything and hence the current design
    datasets.disable_caching()

    # breakpoint()
    dataset, access_func = load_dataset_from_config(step_conf.dataset_in)
    if global_conf.is_v2_run:
        pass
    elif not global_conf.rand3_idx_override:
        set_rand3_idx(step_conf.dataset_in.name)
    else:
        set_rand3_idx(global_conf.rand3_idx_override)
    # breakpoint()
    # gpt_method = get_model(global_conf.model OpenAISynthesis(dataset)
    global gpt_method
    if gpt_method is None:
        gpt_method = LM(global_conf.model.lm_model)

    # breakpoint()
    if step_conf.method in global_conf.transform_dataset.need_unload_gpt:
        gpt_method.unload_model()

    output_path = Path(step_conf.dataset_out.path).parent
    print_and_write_results = partial(
        print_and_write_results_template,
        output_path=output_path / "results.txt",
    )
    print_and_write_results_json = partial(
        print_and_write_results_json_template,
        steps_path=Path(global_conf.transform_dataset.steps_folder),
        step_name=step_conf.exp_name,
    )
    output_dataset_key = step_conf.dataset_out.key

    dataset_name = global_conf.dataset.name

    method_name_config = dict(
        map(
            lambda x: x if len(x) == 2 else (x[0], True),
            map(lambda x: x.split("_", 1), step_conf.method.split("-")),
        )
    )

    exp_name_config = dict(
        map(
            lambda x: x if len(x) == 2 else (x[0], True),
            map(lambda x: x.split("_", 1), step_conf.exp_name.split("-")),
        )
    )

    if step_conf.extra_config is not None and "loadds" in step_conf.extra_config:
        from src.steps.load_entry import process

        dataset = process(gpt_method=gpt_method, **locals())

    elif dataset == STEP_FLAGS.SKIP_CHILD:
        # Shortcircuit if the dataset is skipped
        pass

    elif step_conf.method == "facts":
        from src.steps.facts import process

        dataset = process(gpt_method=gpt_method, **locals())

    elif step_conf.method == "facts-baseline":
        from src.steps.facts import process

        dataset = process(gpt_method=gpt_method, **locals())
    
    elif step_conf.method == "randomize_idx_selection":
        from src.steps.randomize_idx_selection import process

        dataset = process(gpt_method=gpt_method, **locals())

    elif step_conf.method == "load_idx_selection":
        from src.steps.load_idx_selection import process

        dataset = process(gpt_method=gpt_method, **locals())

    elif step_conf.method == "randomize_context":
        from src.steps.randomize_context import process

        dataset = process(gpt_method=gpt_method, **locals())

    elif step_conf.method == "paraphrase_aux":
        from src.steps.paraphrase_aux import process

        dataset = process(gpt_method=gpt_method, **locals())

    elif step_conf.method == "facts-dedup":
        dataset = dataset.map(
            wrap_dataset_map_func(
                lambda x: list(dict.fromkeys(x)), access_func, output_dataset_key
            ),
        )

        set_access_key(dataset, "sanitized_facts", output_dataset_key, global_conf)

    elif step_conf.method == "paraphrase_facts":
        from src.steps.paraphrase_facts import process

        dataset = process(gpt_method=gpt_method, **locals())

    elif step_conf.method == "identity":
        dataset = dataset.map(
            wrap_dataset_map_func(lambda x: x, access_func, output_dataset_key)
        )

    elif step_conf.method.startswith("privasis_abstraction"):
        from src.steps.privasis_abstraction import process

        dataset = process(gpt_method=gpt_method, **locals())

    elif step_conf.method.startswith("dpft_sanitize"):
        from src.steps.dpft_sanitize import process

        dataset = process(gpt_method=gpt_method, **locals())

    elif step_conf.method == "scrub_msft":
        from src.steps.scrub_msft import process

        dataset = process(gpt_method=gpt_method, **locals())

    elif step_conf.method.startswith("paraphrase"):
        from src.steps.paraphrase import process

        dataset = process(gpt_method=gpt_method, **locals())

    elif step_conf.method.startswith("sanitize_and_paraphrase"):
        from src.steps.sanitize_and_paraphrase import process

        dataset = process(gpt_method=gpt_method, **locals())

    elif step_conf.method.startswith("advanced_anonymization"):
        from src.steps.advanced_anonymization import process

        dataset = process(gpt_method=gpt_method, **locals())

    elif step_conf.method.startswith("self_disclosure"):
        from src.steps.self_disclosure import process

        dataset = process(gpt_method=gpt_method, **locals())

    elif step_conf.method == "privacy_leakage_categorization":
        from src.steps.privacy_leakage_categorization import process

        dataset = process(gpt_method=gpt_method, **locals())

    elif step_conf.method.startswith("privacy_leakage_categorization_post_process"):
        from src.steps.privacy_leakage_categorization_post_process import process

        dataset = process(gpt_method=gpt_method, **locals())

    elif step_conf.method.startswith("ensure_id"):
        from src.steps.ensure_id import process

        dataset = process(gpt_method=gpt_method, **locals())

    elif step_conf.method.startswith("cfc"):
        from src.steps.create_fused_cue import process

        dataset = process(gpt_method=gpt_method, **locals())

    elif step_conf.method.startswith("crfc"):
        from src.steps.create_fused_cue import process

        dataset = process(gpt_method=gpt_method, **locals())


    elif step_conf.method.startswith("answer"):
        from src.steps.utility_eval_answer import process

        dataset = process(gpt_method=gpt_method, **locals())

    elif step_conf.method == "utility_eval":
        from src.steps.utility_eval import process

        dataset = process(gpt_method=gpt_method, **locals())

    elif step_conf.method == "mauve":
        import mauve

        p_text = filter(
            lambda x: x, dataset["context"]
        )  # Hardcoded. should create a field with original doc
        q_text = filter(lambda x: x, access_func(dataset))

        # p_text = filter(
        #     lambda x: x, [item for sublist in dataset["facts"] for item in sublist]
        # )  # Hardcoded. should create a field with original doc
        # q_text = filter(
        #     lambda x: x, [item for sublist in access_func(dataset) for item in sublist]
        # )
        out = mauve.compute_mauve(
            p_text=p_text,
            q_text=q_text,
            device_id=0,
            max_text_length=512,
            verbose=False,
            batch_size=32,
        )
        print_and_write_results(f"Mauve score: {out.mauve}")
        return
    
    elif step_conf.method == "count_facts":
        count = 0
        for entry in dataset:
            facts = access_func(entry)
            count += len(facts)
        print_and_write_results(f"average facts: {count / len(dataset)}")
        return

    elif step_conf.method == "rouge":
        from src.steps.rouge import process

        dataset = process(gpt_method=gpt_method, **locals())

    elif step_conf.method.startswith("rouge_matcher"):
        from src.steps.rouge_matcher import process

        dataset = process(gpt_method=gpt_method, **locals())

    elif step_conf.method.startswith("establish_and_search_index"):
        global_conf.datastore.dataset_in = step_conf.dataset_in
        key = global_conf.datastore.dataset_in.key

        if (
            method_name_config is not None
            and method_name_config.get("use", None) is not None
        ):
            if method_name_config["use"] == "grit":
                global_conf.model.datastore_model = global_conf.registry.get("grit")
            elif method_name_config["use"] == "bm25":
                global_conf.model.datastore_model = global_conf.registry.get("bm25")
            else:
                raise NotImplementedError
        else:
            global_conf.model.datastore_model = global_conf.registry.get("contriever")

        from src.embed import generate_passage_embeddings

        logging.info("\n\n************** Building Embedding ***********")
        generate_passage_embeddings(step_conf, global_conf)

        from src.index import build_index

        logging.info("\n\n************** Building Index ***********")
        build_index(global_conf)

        from src.privacy_eval import privacy_eval

        logging.info("\n\n************** Matching ***********")
        dataset = privacy_eval(step_conf, global_conf, dataset, access_func)
        # return

    elif step_conf.method.startswith("privacy_eval"):
        model_name_config = dict(
            map(lambda x: x.split("_", 1), step_conf.method.split("-"))
        )
        privacy_eval_method = "age_string_matching"

        if "use" in model_name_config:
            privacy_eval_method = model_name_config["use"]

        facts_key_orig = get_access_key(dataset, "original_facts", global_conf)
        facts_key_sanitized = get_access_key(dataset, "sanitized_facts", global_conf)
        # facts_key_sanitized = regex_find(
        #     global_conf.evaluation.privacy.samples_key_from_sanitized,
        #     dataset.column_names,
        # )
        context_key_orig = get_access_key(dataset, "original_document", global_conf)
        context_key_sanitized = get_access_key(
            dataset, "sanitized_document", global_conf
        )
        facts_key_matching = get_access_key(
            dataset, global_conf.evaluation.privacy.samples_key_from_orig, global_conf
        )
        # context_key_sanitized = (
        #     "dpft_sanitize-cued_qa-ep_3-eps_3-med_qa_factorized-llama-1000"
        # )
        range_maps = global_conf.range_maps

        slice_generator_func = lambda y: (
            slice(*(map(lambda x: int(x) if x else None, y.split(":"))))
            if ":" in y
            else y
        )

        if global_conf.is_cfc:
            cfc_key = get_access_key(dataset, "matching_cues", global_conf)
            if cfc_key.startswith("fused_facts_"):
                cfc_range = cfc_key.replace("fused_facts_", "")
            elif cfc_key.startswith("paraphrased_fused_facts_"):
                cfc_range = cfc_key.replace("paraphrased_fused_facts_", "")
            else:
                assert False

            range_of_interest_str = parse_create_fused_cue_range(cfc_range, global_conf)
            assert type(range_of_interest_str) == str

            slice_generator_func = lambda y: (
                slice(*(map(lambda x: int(x) if x else None, y.split(":"))))
                if ":" in y
                else y
            )

            ranges = list(map(slice_generator_func, [range_of_interest_str]))

            cfc_range = ranges[0]
        else:
            cfc_key = None
            cfc_range = None

        assert (
            global_conf.is_cfc
            == (cfc_range is not None)
            == bool(cfc_key)
            == ((len(range_maps) == 1) and (range_maps[0] == ":1"))
        )

        ranges = list(map(slice_generator_func, range_maps))

        random.seed(global_conf.seed)

        from src.utils import get_top_article_idx

        if privacy_eval_method == "empty_matching":
            from src.privacy_metrics.empty_matching import process

            # Multiple issues... need to fix
            # need to use a registration decorator
            # need to actually pass in variables
            # need to dynamically load the function rather than hard coding
            dataset = process(gpt_method=gpt_method, **locals())

        elif privacy_eval_method == "correct_matching":
            from src.privacy_metrics.correct_matching import process

            # Multiple issues... need to fix
            # need to use a registration decorator
            # need to actually pass in variables
            # need to dynamically load the function rather than hard coding
            dataset = process(gpt_method=gpt_method, **locals())

        elif privacy_eval_method == "pii_existence":
            from src.privacy_metrics.pii_existence import process

            # Multiple issues... need to fix
            # need to use a registration decorator
            # need to actually pass in variables
            # need to dynamically load the function rather than hard coding
            dataset = process(gpt_method=gpt_method, **locals())

        elif privacy_eval_method == "fact_privacy":
            from src.privacy_metrics.fact_privacy import process

            # Multiple issues... need to fix
            # need to use a registration decorator
            # need to actually pass in variables
            # need to dynamically load the function rather than hard coding
            dataset = process(gpt_method=gpt_method, **locals())

        elif privacy_eval_method == "embed_privacy":
            from src.privacy_metrics.embed_privacy import process

            # Multiple issues... need to fix
            # need to use a registration decorator
            # need to actually pass in variables
            # need to dynamically load the function rather than hard coding
            dataset = process(gpt_method=gpt_method, **locals())

        elif privacy_eval_method == "matched_rouge_privacy":
            from src.privacy_metrics.matched_rouge_privacy import process

            # Multiple issues... need to fix
            # need to use a registration decorator
            # need to actually pass in variables
            # need to dynamically load the function rather than hard coding
            dataset = process(gpt_method=gpt_method, **locals())

        elif privacy_eval_method == "aux_n_gram_leakage":
            from privacy_metrics.aux_n_gram_leakage import process

            # Multiple issues... need to fix
            # need to use a registration decorator
            # need to actually pass in variables
            # need to dynamically load the function rather than hard coding
            dataset = process(gpt_method=gpt_method, **locals())

        if not dataset:
            return

        return

    else:
        raise NotImplementedError(f"Step {step_conf.method} not implemented")

    if dataset == STEP_FLAGS.SKIP_CHILD:
        skipped_file = Path(step_conf.dataset_out.path) / "SKIPPED"
        skipped_file.mkdir(parents=True, exist_ok=True)
        skipped_file.touch()
        return

    if dataset is None:
        return

    if check_contain_pending_batched_output(dataset):
        raise "Pending batched output found. Resolve it before saving the dataset"

    if step_conf.get("force_write", False):
        step_conf.dataset_out = step_conf.dataset_in
        shutil.rmtree(step_conf.dataset_out.path)
        dataset.save_to_disk(step_conf.dataset_out.path + "_1")
        shutil.move(step_conf.dataset_out.path + "_1", step_conf.dataset_out.path)
    else:
        dataset.save_to_disk(step_conf.dataset_out.path)

    df = dataset.to_pandas()
    # For inspection purposes
    df.head(10).to_json(
        Path(step_conf.dataset_out.path) / "inspect.json", orient="records"
    )

    if step_conf.dataset_out.get("save_as_csv", False):
        df.to_csv(Path(step_conf.dataset_out.path) / "data.csv", index=False)
