import hydra
import logging
from omegaconf.omegaconf import OmegaConf, open_dict
from pathlib import Path

from transform_dataset import transform_dataset

# PYTHONPATH=/mmfs1/home/rx31/projects/privacy-pipeline python src/run.py
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mmfs1/gscratch/sewoong/rx31/miniconda3/envs/privacy-pipeline/lib

# Name entry
OmegaConf.register_new_resolver("ne", lambda x: f"-{x}" if x else "")
OmegaConf.register_new_resolver(
    "method_list2str",
    lambda x: "-".join(reversed(list(filter(lambda x: type(x) == str, x)))),
)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg):
    if cfg.get("override_config_path", None):
        logging.info("\n\n************** Overriding configuration ***********")
        logging.info(f"New Location: {cfg.override_config_path}")

        cfg = OmegaConf.load(cfg.override_config_path)

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    steps_folder = Path(cfg.transform_dataset.steps_folder)
    # doing parents here for checkpoints
    steps_folder.mkdir(parents=True, exist_ok=True)
    with open(steps_folder / "steps_config.yaml", "w") as f:
        OmegaConf.save(config=cfg, f=f)

    # For historical reasons, this is not really necessary as I've moved all tasks into the final file
    if cfg.tasks.get("transform_dataset", False):
        logging.info("\n\n************** Preparing Dataset ***********")
        transform_dataset(cfg)

    # if cfg.tasks.datastore.get('embedding', False):
    #     from src.embed import generate_passage_embeddings
    #     logging.info("\n\n************** Building Embedding ***********")
    #     generate_passage_embeddings(cfg)

    # if cfg.tasks.datastore.get('index', False):
    #     from src.index import build_index
    #     logging.info("\n\n************** Building Index ***********")
    #     build_index(cfg)

    # if cfg.tasks.evaluation.get('privacy', False):
    #     from src.privacy_eval import privacy_eval
    #     logging.info("\n\n************** Privacy Eval ***********")
    #     privacy_eval(cfg)


if __name__ == "__main__":
    main()
