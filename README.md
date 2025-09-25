# Privacy Pipeline


## Installation

### Prerequisites
- Python 3.10
- `uv` (Python package manager)

### Installing uv
First, install `uv` [here](https://docs.astral.sh/uv/getting-started/installation/).

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd privacy-pipeline
```

2. Install dependencies using uv:
```bash
uv sync
```

## Quick Start

### Basic Usage
Run the main pipeline with a predefined dataset:
```bash
uv run src/run.py +dataset=[medqa,1k]
```

The pipeline includes dataset transformation capabilities through the `transform_dataset` module, which handles:
- Data sanitization and anonymization
- Privacy evaluation metrics  
- Utility preservation analysis

### Available Datasets
- `medqa`
- `wildchat`

### Dataset Configurations
- `1k` - Limited subset (which is used in the paper)
- `all` - Full dataset

## Configuration

The pipeline uses Hydra for configuration management. Main configuration files:

- `configs/config.yaml` - Main pipeline configuration
- `configs/dataset/` - Dataset-specific configurations
- `configs/registry.yaml` - Model and method registry

### Key Configuration Options

#### Sanitization Methods
Configure the sanitization approach in `configs/config.yaml`:
```yaml
sanitization:
  step: sanitize_and_paraphrase-use_gpt  # or other methods
```

Available sanitization steps:
- `sanitize_and_paraphrase-use_gpt`
- `advanced_anonymization`
- `self_disclosure` - requires setup
- `dpft_sanitize` - requires a trained model 
- `scrub_msft` - requires setting up the api

## Creating Custom Datasets

1. **Create a dataset configuration file** following the pattern in `configs/dataset/medqa.yaml`:
```yaml
name: your_dataset_name
load_method: load_datasets_json  # or other loading method
path: data/your_data.jsonl
key: context                     # field containing the text data
remove_last_line: false
count: ""                       # leave empty for full dataset
split: train
split_arg: ${.split}[:${.count}]
```

2. **Place your data file** in the `data/` directory

3. **Run the pipeline** with your new dataset:
```bash
uv run src/run.py +dataset=your_dataset_name
```

## Adding Custom Transformation Functions

### Adding Custom Sanitization Methods

1. **Create a new sanitization script** in `src/steps/` following the template of `src/steps/sanitize_and_paraphrase.py`

2. **Implement the required `process` function**:
```python
def process(step_conf, global_conf, dataset, access_func, output_path, 
           print_and_write_results, print_and_write_results_json, 
           output_dataset_key, gpt_method, **kwargs):
    # Your sanitization logic here
    return processed_dataset
```

3. **Register your method** in `src/transform_dataset.py` by adding an `elif` condition:
```python
elif step_conf.method.startswith("your_method_name"):
    from src.steps.your_method_script import process
    dataset = process(step_conf, global_conf, dataset, access_func, 
                     output_path, print_and_write_results, 
                     print_and_write_results_json, output_dataset_key, 
                     gpt_method, **kwargs)
```

4. **Update the configuration** to use your new method:
```yaml
transform_dataset:
  method: 
    - your_method_name
```
