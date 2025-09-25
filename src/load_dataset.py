from functools import partial
from datasets import load_dataset, Dataset, load_from_disk
from pathlib import Path
from src.utils import STEP_FLAGS

def remove_last_sentence(text):
    # Split the text into sentences
    sentences = text.split('. ')

    # Remove the last sentence if there are multiple sentences
    if len(sentences) > 1:
        last_sentence = sentences.pop()
        cleaned_text = '. '.join(sentences) + '. '
        return cleaned_text
    else:
        return text  # Return the original text if there's only one sentence

def access_func_template(cfg, x):
    if cfg.get('remove_last_line', False):
        x = remove_last_sentence(x)
    return x[cfg.key]

def load_dataset_from_config(cfg):
    if (Path(cfg.path) / 'SKIPPED').exists():
        print(f"Skipping {cfg.path}")
        return STEP_FLAGS.SKIP_CHILD, None

    load_method = cfg.get('load_method', 'load_from_disk')
    if load_method == 'load_datasets':
        dataset = load_dataset(cfg.path, split=cfg.split_arg)

        # When loading from huggingface hub or json, we should ensure that count is None
        count = cfg.get('count', None)
        assert count is None
    if load_method == 'load_datasets_json':
        # Warning: age of -1 are processed to "-1". Need to 
        dataset = load_dataset("json", data_files=cfg.path, split=cfg.split)

        shuffle = cfg.get('shuffle', None)
        if shuffle:
            dataset = dataset.shuffle(seed=shuffle)
        count = cfg.get('count', None)
        if count:
            dataset = Dataset.from_dict(dataset[:count])
        # count = cfg.get('count', None)
        # assert count is None
    if load_method == 'load_from_disk':
        # Warning: age of -1 are processed to "-1". Need to 
        dataset = load_from_disk(cfg.path)
        split = cfg.get('split', None)
        if split:
            dataset = dataset[split]
        shuffle = cfg.get('shuffle', None)
        if shuffle:
            dataset = dataset.shuffle(seed=shuffle)
        count = cfg.get('count', None)
        if count:
            dataset = Dataset.from_dict(dataset[:count])
    
    access_func = partial(access_func_template, cfg)

    return dataset, access_func